# -*- coding: utf-8 -*-
from __future__ import print_function

import argparse
import os
import time
import random
import numpy as np
import tensorflow as tf

from net import *  # Network, data_loader_bd_rm_from_tfrecord, GPU_ID などを想定
from scores import fast_hist
from utils.rgb_ind_convertor import (
    ind2rgb,
    rgb2ind,
    floorplan_boundary_map,
    floorplan_fuse_map,
)

try:
    # scipy.misc は古い環境向け（元実装準拠）
    from scipy.misc import imread, imresize, imsave
except Exception:
    # scipy.misc がない環境へのフォールバック（可能なら）
    import imageio

    def imread(path, mode="RGB"):
        img = imageio.imread(path)
        if mode == "L" and img.ndim == 3:
            # 簡易グレースケール化
            img = (0.299 * img[..., 0] + 0.587 * img[..., 1] + 0.114 * img[..., 2]).astype(
                np.uint8
            )
        return img

    def imresize(img, size):
        import cv2

        if isinstance(size, tuple) and len(size) == 2:
            # (H, W) 指定
            return cv2.resize(img, (size[1], size[0]), interpolation=cv2.INTER_NEAREST)
        elif isinstance(size, tuple) and len(size) == 3:
            # (H, W, C) の場合は空間のみ
            return cv2.resize(img, (size[1], size[0]), interpolation=cv2.INTER_NEAREST)
        else:
            raise ValueError("Unsupported size format for imresize fallback")

    def imsave(path, img):
        import imageio

        imageio.imwrite(path, img)

# --- 実行環境設定（元実装準拠） ---
os.environ["CUDA_VISIBLE_DEVICES"] = GPU_ID  # net.py で定義されている想定
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

seed = 8964

# --- 引数 ---
parser = argparse.ArgumentParser()
parser.add_argument("--phase", type=str, default="Test", help="Train/Test network.")


# ---- ヘルパー（evaluate の安全化用） ----
def _squeeze_hw1(x):
    """(H,W,1)->(H,W) に潰す。その他はそのまま返す。"""
    x = np.asarray(x)
    if x.ndim == 3 and x.shape[-1] == 1:
        return x[..., 0]
    return x


def _ensure_same_hw(a, b):
    """a を b の (H,W) と一致させる（最近傍）。"""
    a = np.asarray(a)
    b = np.asarray(b)
    if a.shape == b.shape:
        return a
    import cv2

    return cv2.resize(a, (b.shape[1], b.shape[0]), interpolation=cv2.INTER_NEAREST)


class MODEL(Network):
    """Training / Inference / Evaluation wrapper"""

    def __init__(self):
        Network.__init__(self)
        self.log_dir = "pretrained"
        self.eval_file = "./dataset/r3d_test.txt"
        self.loss_type = "balanced"

    def convert_one_hot_to_image(self, one_hot, dtype="float", act=None):
        if act == "softmax":
            one_hot = tf.nn.softmax(one_hot, axis=-1)
        n, h, w, c = one_hot.shape.as_list()
        im = tf.reshape(tf.argmax(one_hot, axis=-1), [n, h, w, 1])
        if dtype == "int":
            im = tf.cast(im, dtype=tf.uint8)
        else:
            im = tf.cast(im, dtype=tf.float32)
        return im

    def cross_two_tasks_weight(self, y1, y2):
        p1 = tf.reduce_sum(y1)
        p2 = tf.reduce_sum(y2)
        w1 = p2 / (p1 + p2)
        w2 = p1 / (p1 + p2)
        return w1, w2

    def balanced_entropy(self, x, y):
        # cliped_by_eps
        eps = 1e-6
        z = tf.nn.softmax(x)
        cliped_z = tf.clip_by_value(z, eps, 1 - eps)
        log_z = tf.math.log(cliped_z)
        num_classes = y.shape.as_list()[-1]

        ind = tf.argmax(y, -1, output_type=tf.int32)
        total = tf.reduce_sum(y)

        # 各クラスのマスク & ピクセル数
        m_c = []
        n_c = []
        for c in range(num_classes):
            m_c.append(tf.cast(tf.equal(ind, c), dtype=tf.int32))
            n_c.append(tf.cast(tf.reduce_sum(m_c[-1]), dtype=tf.float32))

        # 重み計算
        c = []
        for i in range(num_classes):
            c.append(total - n_c[i])
        tc = tf.add_n(c)

        loss = 0.0
        for i in range(num_classes):
            w = c[i] / tc
            m_c_one_hot = tf.one_hot((i * m_c[i]), num_classes, axis=-1)
            y_c = m_c_one_hot * y
            loss += w * tf.reduce_mean(-tf.reduce_sum(y_c * log_z, axis=1))
        return loss / num_classes  # mean

    def train(self, loader_dict, num_batch, max_step=40000):
        images = loader_dict["images"]
        labels_r_hot = loader_dict["label_rooms"]
        labels_cw_hot = loader_dict["label_boundaries"]

        max_ep = max_step // num_batch
        print("max_step = {}, max_ep = {}, num_batch = {}".format(max_step, max_ep, num_batch))

        logits1, logits2 = self.forward(images, init_with_pretrain_vgg=False)

        if self.loss_type == "balanced":
            # in-task balance
            loss1 = self.balanced_entropy(logits1, labels_r_hot)
            loss2 = self.balanced_entropy(logits2, labels_cw_hot)
        else:
            loss1 = tf.reduce_mean(
                tf.nn.softmax_cross_entropy_with_logits_v2(
                    logits=logits1, labels=labels_r_hot, name="bce1"
                )
            )
            loss2 = tf.reduce_mean(
                tf.nn.softmax_cross_entropy_with_logits_v2(
                    logits=logits2, labels=labels_cw_hot, name="bce2"
                )
            )

        # cross-task balance
        w1, w2 = self.cross_two_tasks_weight(labels_r_hot, labels_cw_hot)
        loss = (w1 * loss1 + w2 * loss2)

        optim = tf.train.AdamOptimizer(learning_rate=1e-4).minimize(
            loss, colocate_gradients_with_ops=True
        )

        # セッション
        config = tf.ConfigProto(allow_soft_placement=True)
        config.gpu_options.allow_growth = True
        with tf.Session(config=config) as sess:
            sess.run(tf.group(tf.global_variables_initializer(), tf.local_variables_initializer()))
            saver = tf.train.Saver(max_to_keep=10)

            coord = tf.train.Coordinator()
            threads = tf.train.start_queue_runners(sess=sess, coord=coord)

            print("Start Training!")
            total_times = 0.0
            for ep in range(max_ep):  # epoch
                for n in range(num_batch):  # batch
                    tic = time.time()
                    _loss, _ = sess.run([loss, optim])
                    duration = time.time() - tic
                    total_times += duration
                    step = int(ep * num_batch + n)
                    print(
                        "step {}: loss = {:.3f}; {:.2f} data/sec, executed {} minutes".format(
                            step, _loss, 1.0 / max(duration, 1e-9), int(total_times / 60)
                        )
                    )

                # 2epoch ごとに保存＋評価
                if ep % 2 == 0:
                    saver.save(sess, self.log_dir + "/model", global_step=ep)
                    self.evaluate(sess=sess, epoch=ep)

            saver.save(sess, self.log_dir + "/model", global_step=max_ep)
            self.evaluate(sess=sess, epoch=max_ep)

            coord.request_stop()
            coord.join(threads)
            sess.close()

    def infer(self, save_dir="out", resize=True, merge=True):
        print("generating test set of {}....".format(self.eval_file))
        print("will save to [./{}]".format(save_dir))

        room_dir = os.path.join(save_dir, "room")
        close_wall_dir = os.path.join(save_dir, "boundary")
        if not os.path.exists(save_dir):
            os.mkdir(save_dir)
        if not os.path.exists(room_dir):
            os.mkdir(room_dir)
        if not os.path.exists(close_wall_dir):
            os.mkdir(close_wall_dir)

        x = tf.placeholder(shape=[1, 512, 512, 3], dtype=tf.float32)
        logits1, logits2 = self.forward(x, init_with_pretrain_vgg=False)
        rooms = self.convert_one_hot_to_image(logits1, act="softmax", dtype="int")
        close_walls = self.convert_one_hot_to_image(logits2, act="softmax", dtype="int")

        config = tf.ConfigProto(allow_soft_placement=True)
        sess = tf.Session(config=config)
        sess.run(tf.group(tf.global_variables_initializer(), tf.local_variables_initializer()))
        saver = tf.train.Saver()
        saver.restore(sess, save_path=tf.train.latest_checkpoint(self.log_dir))

        # 一枚ずつ推論
        paths = open(self.eval_file, "r").read().splitlines()
        paths = [p.split("\t")[0] for p in paths]
        for p in paths:
            im = imread(p, mode="RGB")
            im_x = imresize(im, (512, 512, 3)) / 255.0  # resize + normalize
            im_x = np.reshape(im_x, (1, 512, 512, 3))

            out1, out2 = sess.run([rooms, close_walls], feed_dict={x: im_x})

            if resize:
                out1_rgb = ind2rgb(np.squeeze(out1))
                out1_rgb = imresize(out1_rgb, (im.shape[0], im.shape[1]))
                out2_rgb = ind2rgb(np.squeeze(out2), color_map=floorplan_boundary_map)
                out2_rgb = imresize(out2_rgb, (im.shape[0], im.shape[1]))
            else:
                out1_rgb = ind2rgb(np.squeeze(out1))
                out2_rgb = ind2rgb(np.squeeze(out2), color_map=floorplan_boundary_map)

            if merge:
                out1i = np.squeeze(out1)
                out2i = np.squeeze(out2)
                out1i[out2i == 2] = 10
                out1i[out2i == 1] = 9
                out3_rgb = ind2rgb(out1i, color_map=floorplan_fuse_map)
            # 保存
            name = p.split("/")[-1]
            save_path1 = os.path.join(room_dir, name.split(".jpg")[0] + "_rooms.png")
            save_path2 = os.path.join(close_wall_dir, name.split(".jpg")[0] + "_bd_rm.png")
            save_path3 = os.path.join(save_dir, name.split(".jpg")[0] + "_rooms.png")
            imsave(save_path1, out1_rgb)
            imsave(save_path2, out2_rgb)
            if merge:
                imsave(save_path3, out3_rgb)
            print("Saving prediction: {}".format(name))

    def evaluate(self, sess, epoch, num_of_classes=11):
        x = tf.placeholder(shape=[1, 512, 512, 3], dtype=tf.float32)
        logits1, logits2 = self.forward(x, init_with_pretrain_vgg=False)
        predict_bd = self.convert_one_hot_to_image(logits2, act="softmax", dtype="int")
        predict_room = self.convert_one_hot_to_image(logits1, act="softmax", dtype="int")

        paths = open(self.eval_file, "r").read().splitlines()
        image_paths = [p.split("\t")[0] for p in paths]  # image
        gt2_paths = [p.split("\t")[2] for p in paths]  # 2: doors & windows
        gt3_paths = [p.split("\t")[3] for p in paths]  # 3: rooms
        gt4_paths = [p.split("\t")[-1] for p in paths]  # last: close wall

        n = len(paths)
        hist = np.zeros((num_of_classes, num_of_classes))

        for i in range(n):
            # 入力画像
            im = imread(image_paths[i], mode="RGB")
            im = imresize(im, (512, 512, 3)) / 255.0
            im = np.reshape(im, (1, 512, 512, 3))

            # ---- GT の読み込み・整形 ----
            # 部屋（RGB → インデックス）
            rr = imread(gt3_paths[i], mode="RGB")
            rr = imresize(rr, (512, 512, 3))
            rr_ind = rgb2ind(rr)

            # 壁の閉領域（L）
            cw = imread(gt4_paths[i], mode="L")
            cw = imresize(cw, (512, 512)) / 255.0
            cw = (cw > 0.5).astype(np.uint8)

            # ドア・窓（L）
            dd = imread(gt2_paths[i], mode="L")
            dd = imresize(dd, (512, 512)) / 255.0
            dd = (dd > 0.5).astype(np.uint8)

            # --- ここから安全化（IndexError 対策） ---
            rr_ind = _squeeze_hw1(rr_ind)
            dd = _squeeze_hw1(dd)
            cw = _squeeze_hw1(cw)

            # 型とサイズを揃える
            if dd.dtype != np.uint8:
                dd = (dd > 0.5).astype(np.uint8)
            if cw.dtype != np.uint8:
                cw = (cw > 0.5).astype(np.uint8)

            dd = _ensure_same_hw(dd, rr_ind)
            cw = _ensure_same_hw(cw, rr_ind)

            # rr_ind が (H,W,3) で来る可能性に備える（基本は (H,W)）
            if rr_ind.ndim == 3 and rr_ind.shape[-1] == 3:
                # あり得ない想定だが念のため：RGB を index 化
                rr_ind = rgb2ind(rr_ind)

            # マージ（GT）
            rr_ind[cw == 1] = 10
            rr_ind[dd == 1] = 9
            # --- 安全化ここまで ---

            # 予測
            rm_ind, bd_ind = sess.run([predict_room, predict_bd], feed_dict={x: im})
            rm_ind = np.squeeze(rm_ind)
            bd_ind = np.squeeze(bd_ind)

            # マージ（予測）
            rm_ind[bd_ind == 2] = 10
            rm_ind[bd_ind == 1] = 9

            # スコア加算
            hist += fast_hist(rm_ind.flatten(), rr_ind.flatten(), num_of_classes)

        overall_acc = np.diag(hist).sum() / hist.sum()
        mean_acc = np.diag(hist) / (hist.sum(1) + 1e-6)
        mean_acc9 = (np.nansum(mean_acc[:7]) + mean_acc[-2] + mean_acc[-1]) / 9.0

        with open("EVAL_" + self.log_dir, "a") as f:
            print(
                "Model at epoch {}: overall accuracy = {:.4f}, mean_acc = {:.4f}".format(
                    epoch, overall_acc, mean_acc9
                ),
                file=f,
            )
            for i in range(mean_acc.shape[0]):
                if i not in [7, 8]:  # ignore class 7 & 8
                    print(
                        "\t\tepoch {}: {}th label: accuracy = {:.4f}".format(
                            epoch, i, mean_acc[i]
                        ),
                        file=f,
                    )


def main(args):
    tf.random.set_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    model = MODEL()

    if args.phase.lower() == "train":
        loader_dict, num_batch = data_loader_bd_rm_from_tfrecord(batch_size=1)
        tic = time.time()
        model.train(loader_dict, num_batch)
        toc = time.time()
        print("total training + evaluation time = {} minutes".format((toc - tic) / 60.0))
    elif args.phase.lower() == "test":
        model.infer()
    else:
        pass


if __name__ == "__main__":
    FLAGS, unparsed = parser.parse_known_args()
    main(FLAGS)
