# -*- coding: utf-8 -*-
"""
DeepFloorplan main.py — Python3 / TF2互換(compat.v1) 版
- print文 → print()
- xrange → range
- tf.log → tf.math.log
- tf.Session / Saver 等は tf.compat.v1.*
- TF2 環境で eager を無効化（disable_eager_execution）
"""

from __future__ import print_function
import os
import time
import random
import argparse
import numpy as np
import tensorflow as tf

# 画像入出力（元コードの imread/imsave 相当）
from imageio.v2 import imread, imwrite as imsave

# リポジトリ内ユーティリティ一式
# - Network, data_loader_bd_rm_from_tfrecord, imresize, ind2rgb, rgb2ind,
#   floorplan_boundary_map, floorplan_fuse_map, fast_hist などを想定
from net import *

# ===== TF2 → TF1 互換モード =====
tf.compat.v1.disable_eager_execution()

# ===== GPU設定（net.py 等で GPU_ID が定義されている場合に従う、無ければ '0'）=====
try:
    _GPU_ID = GPU_ID  # from net import *
except Exception:
    _GPU_ID = '0'
os.environ['CUDA_VISIBLE_DEVICES'] = str(_GPU_ID)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

seed = 8964

# ===== 引数 =====
parser = argparse.ArgumentParser()
parser.add_argument('--phase', type=str, default='Test', help='Train/Test network.')

class MODEL(Network):
    """DeepFloorplan model wrapper"""
    def __init__(self):
        super(MODEL, self).__init__()
        self.log_dir = 'pretrained'
        self.eval_file = './dataset/r3d_test.txt'
        self.loss_type = 'balanced'

    def convert_one_hot_to_image(self, one_hot, dtype='float', act=None):
        if act == 'softmax':
            one_hot = tf.nn.softmax(one_hot, axis=-1)

        n, h, w, c = one_hot.shape.as_list()
        im = tf.reshape(tf.argmax(one_hot, axis=-1), [n, h, w, 1])

        if dtype == 'int':
            im = tf.cast(im, dtype=tf.uint8)
        else:
            im = tf.cast(im, dtype=tf.float32)
        return im

    def cross_two_tasks_weight(self, y1, y2):
        p1 = tf.reduce_sum(y1)
        p2 = tf.reduce_sum(y2)
        w1 = p2 / (p1 + p2 + 1e-12)
        w2 = p1 / (p1 + p2 + 1e-12)
        return w1, w2

    def balanced_entropy(self, x, y):
        """マルチクラスの頻度バランスをとったクロスエントロピー"""
        eps = 1e-6
        z = tf.nn.softmax(x)
        cliped_z = tf.clip_by_value(z, eps, 1 - eps)
        log_z = tf.math.log(cliped_z)

        num_classes = y.shape.as_list()[-1]
        ind = tf.argmax(y, -1, output_type=tf.int32)

        total = tf.reduce_sum(y)  # 総前景画素

        m_c = []  # 各クラスのインデックスマスク
        n_c = []  # 各クラスの画素数
        for c in range(num_classes):
            m = tf.cast(tf.equal(ind, c), dtype=tf.int32)
            m_c.append(m)
            n_c.append(tf.cast(tf.reduce_sum(m), dtype=tf.float32))

        # 他クラス総和 c[i] = total - n_c[i]
        c_list = []
        for i in range(num_classes):
            c_list.append(total - n_c[i])
        tc = tf.add_n(c_list) + 1e-12

        loss = 0.0
        for i in range(num_classes):
            w = c_list[i] / tc
            m_c_one_hot = tf.one_hot((i * m_c[i]), num_classes, axis=-1)
            y_c = m_c_one_hot * y
            # バッチ平均
            loss += w * tf.reduce_mean(-tf.reduce_sum(y_c * log_z, axis=1))

        return loss / float(num_classes)

    def train(self, loader_dict, num_batch, max_step=40000):
        images = loader_dict['images']
        labels_r_hot = loader_dict['label_rooms']
        labels_cw_hot = loader_dict['label_boundaries']

        max_ep = max_step // max(1, num_batch)
        print('max_step = {}, max_ep = {}, num_batch = {}'.format(max_step, max_ep, num_batch))

        logits1, logits2 = self.forward(images, init_with_pretrain_vgg=False)

        if self.loss_type == 'balanced':
            loss1 = self.balanced_entropy(logits1, labels_r_hot)
            loss2 = self.balanced_entropy(logits2, labels_cw_hot)
        else:
            # 互換：_v2 は TF2 では非推奨。compat でも可
            loss1 = tf.reduce_mean(
                tf.nn.softmax_cross_entropy_with_logits(labels=labels_r_hot, logits=logits1, name='bce1')
            )
            loss2 = tf.reduce_mean(
                tf.nn.softmax_cross_entropy_with_logits(labels=labels_cw_hot, logits=logits2, name='bce2')
            )

        # タスク間重み
        w1, w2 = self.cross_two_tasks_weight(labels_r_hot, labels_cw_hot)
        loss = (w1 * loss1 + w2 * loss2)

        optim = tf.compat.v1.train.AdamOptimizer(learning_rate=1e-4).minimize(
            loss, colocate_gradients_with_ops=True
        )

        # ===== Session 構築 =====
        config = tf.compat.v1.ConfigProto(allow_soft_placement=True)
        config.gpu_options.allow_growth = True

        with tf.compat.v1.Session(config=config) as sess:
            sess.run(tf.group(tf.compat.v1.global_variables_initializer(),
                              tf.compat.v1.local_variables_initializer()))

            saver = tf.compat.v1.train.Saver(max_to_keep=10)

            coord = tf.compat.v1.train.Coordinator()
            threads = tf.compat.v1.train.start_queue_runners(sess=sess, coord=coord)

            print("Start Training!")
            total_times = 0.0

            for ep in range(int(max_ep)):  # epoch
                for n in range(int(num_batch)):  # batch
                    tic = time.time()
                    _loss, _ = sess.run([loss, optim])
                    duration = time.time() - tic
                    total_times += duration

                    step = int(ep * num_batch + n)
                    speed = (1.0 / duration) if duration > 0 else 0.0
                    print('step {}: loss = {:.3f}; {:.2f} data/sec, executed {} minutes'.format(
                        step, float(_loss), speed, int(total_times / 60))
                    )

                # 2エポック毎に保存 & 評価
                if ep % 2 == 0:
                    saver.save(sess, os.path.join(self.log_dir, 'model'), global_step=ep)
                    self.evaluate(sess=sess, epoch=ep)

            saver.save(sess, os.path.join(self.log_dir, 'model'), global_step=int(max_ep))
            self.evaluate(sess=sess, epoch=int(max_ep))

            coord.request_stop()
            coord.join(threads)

    def infer(self, save_dir='out', resize=True, merge=True):
        print("generating test set of {}.... will save to [./{}]".format(self.eval_file, save_dir))
        room_dir = os.path.join(save_dir, 'room')
        close_wall_dir = os.path.join(save_dir, 'boundary')

        os.makedirs(save_dir, exist_ok=True)
        os.makedirs(room_dir, exist_ok=True)
        os.makedirs(close_wall_dir, exist_ok=True)

        x = tf.compat.v1.placeholder(shape=[1, 512, 512, 3], dtype=tf.float32)

        logits1, logits2 = self.forward(x, init_with_pretrain_vgg=False)
        rooms = self.convert_one_hot_to_image(logits1, act='softmax', dtype='int')
        close_walls = self.convert_one_hot_to_image(logits2, act='softmax', dtype='int')

        config = tf.compat.v1.ConfigProto(allow_soft_placement=True)
        sess = tf.compat.v1.Session(config=config)
        sess.run(tf.group(tf.compat.v1.global_variables_initializer(),
                          tf.compat.v1.local_variables_initializer()))

        saver = tf.compat.v1.train.Saver()
        saver.restore(sess, save_path=tf.train.latest_checkpoint(self.log_dir))

        paths = open(self.eval_file, 'r').read().splitlines()
        paths = [p.split('\t')[0] for p in paths]
        for p in paths:
            im = imread(p)
            im_x = imresize(im, (512, 512, 3)) / 255.0
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
                o1 = np.squeeze(out1)
                o2 = np.squeeze(out2)
                o1[o2 == 2] = 10
                o1[o2 == 1] = 9
                out3_rgb = ind2rgb(o1, color_map=floorplan_fuse_map)

            name = os.path.basename(p)
            stem = os.path.splitext(name)[0]
            save_path1 = os.path.join(room_dir, f'{stem}_rooms.png')
            save_path2 = os.path.join(close_wall_dir, f'{stem}_bd_rm.png')
            save_path3 = os.path.join(save_dir, f'{stem}_rooms.png')

            imsave(save_path1, out1_rgb)
            imsave(save_path2, out2_rgb)
            if merge:
                imsave(save_path3, out3_rgb)

            print('Saving prediction: {}'.format(name))

    def evaluate(self, sess, epoch, num_of_classes=11):
        x = tf.compat.v1.placeholder(shape=[1, 512, 512, 3], dtype=tf.float32)
        logits1, logits2 = self.forward(x, init_with_pretrain_vgg=False)
        predict_bd = self.convert_one_hot_to_image(logits2, act='softmax', dtype='int')
        predict_room = self.convert_one_hot_to_image(logits1, act='softmax', dtype='int')

        paths = open(self.eval_file, 'r').read().splitlines()
        image_paths = [p.split('\t')[0] for p in paths]
        gt2_paths = [p.split('\t')[2] for p in paths]  # doors/windows
        gt3_paths = [p.split('\t')[3] for p in paths]  # rooms (RGB)
        gt4_paths = [p.split('\t')[-1] for p in paths]  # close wall

        n = len(paths)
        hist = np.zeros((num_of_classes, num_of_classes), dtype=np.float64)

        for i in range(n):
            im = imread(image_paths[i])

            dd = imread(gt2_paths[i])  # L
            rr = imread(gt3_paths[i])  # RGB
            cw = imread(gt4_paths[i])  # L

            im_n = imresize(im, (512, 512, 3)) / 255.0
            im_n = np.reshape(im_n, (1, 512, 512, 3))

            rr = imresize(rr, (512, 512, 3))
            rr_ind = rgb2ind(rr)

            cw = imresize(cw, (512, 512)) / 255.0
            dd = imresize(dd, (512, 512)) / 255.0
            cw = (cw > 0.5).astype(np.uint8)
            dd = (dd > 0.5).astype(np.uint8)

            rr_ind[cw == 1] = 10
            rr_ind[dd == 1] = 9

            rm_ind, bd_ind = sess.run([predict_room, predict_bd], feed_dict={x: im_n})
            rm_ind = np.squeeze(rm_ind)
            bd_ind = np.squeeze(bd_ind)

            rm_ind[bd_ind == 2] = 10
            rm_ind[bd_ind == 1] = 9

            hist += fast_hist(rm_ind.flatten(), rr_ind.flatten(), num_of_classes)

        overall_acc = np.diag(hist).sum() / (hist.sum() + 1e-12)
        mean_acc = np.diag(hist) / (hist.sum(1) + 1e-6)

        # クラス7,8（ignore）を除外した9クラス平均
        valid_idxs = [i for i in range(num_of_classes) if i not in [7, 8]]
        mean_acc9 = np.nanmean(mean_acc[valid_idxs])

        os.makedirs('.', exist_ok=True)
        with open('EVAL_' + self.log_dir, 'a') as f:
            f.write('Model at epoch {}: overall accuracy = {:.4f}, mean_acc = {:.4f}\n'
                    .format(epoch, overall_acc, mean_acc9))
            for i in range(mean_acc.shape[0]):
                if i not in [7, 8]:
                    f.write('\t\tepoch {}: {}th label: accuracy = {:.4f}\n'
                            .format(epoch, i, mean_acc[i]))

def main(args):
    tf.compat.v1.set_random_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    model = MODEL()

    if args.phase.lower() == 'train':
        loader_dict, num_batch = data_loader_bd_rm_from_tfrecord(batch_size=1)
        tic = time.time()
        model.train(loader_dict, num_batch)
        toc = time.time()
        print('total training + evaluation time = {} minutes'.format((toc - tic) / 60.0))
    elif args.phase.lower() == 'test':
        model.infer()
    else:
        print("Unknown phase: {} (use Train/Test)".format(args.phase))

if __name__ == '__main__':
    FLAGS, unparsed = parser.parse_known_args()
    main(FLAGS)
