# demo.py ーー 完全置換版（1/3/4chを自動吸収・保存出力）
import os
import argparse
import numpy as np

# ===== Matplotlib: 非GUIバックエンドで保存 =====
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt

from imageio.v2 import imread, imwrite as imsave
from PIL import Image

# ===== TF1 互換 =====
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
from tensorflow.compat.v1 import flags
FLAGS = flags.FLAGS

# ========== 互換 imresize ==========
def imresize(img, size, interp='bilinear', mode=None):
    """
    Legacy-friendly imresize:
      - size: float scale, (H, W), or (H, W, C)  ※Cは無視
      - float入力のレンジを自動判定:
          * max<=1.0 → 0..1 を返す
          * それ以外 → 0..255 を返す
    """
    import numpy as np
    from PIL import Image

    orig_dtype = img.dtype
    # ---- floatレンジを推定 ----
    is_float = np.issubdtype(orig_dtype, np.floating)
    if is_float:
        maxv = float(np.nanmax(img)) if img.size else 0.0
        float_is_01 = maxv <= 1.0 + 1e-6
    else:
        float_is_01 = False

    # ---- PILへ（常にuint8で渡す）----
    if is_float:
        if float_is_01:
            src = (np.clip(img, 0.0, 1.0) * 255.0).astype(np.uint8)
        else:
            src = np.clip(img, 0.0, 255.0).astype(np.uint8)
    else:
        src = np.clip(img, 0, 255).astype(np.uint8)

    pil_img = Image.fromarray(src)

    # ---- 目標サイズ ----
    if isinstance(size, (int, float)):
        target = (int(round(pil_img.width * float(size))),
                  int(round(pil_img.height * float(size))))
    elif isinstance(size, (tuple, list)):
        if len(size) == 2:
            h, w = int(size[0]), int(size[1])
        elif len(size) == 3:
            h, w = int(size[0]), int(size[1])  # Cは無視
        else:
            raise ValueError(f"Invalid size for imresize: {size}")
        target = (w, h)  # PILは(W,H)
    else:
        raise ValueError(f"Invalid size for imresize: {size}")

    _imap = {
        'nearest': Image.NEAREST,
        'bilinear': Image.BILINEAR,
        'bicubic': Image.BICUBIC,
        'lanczos': Image.LANCZOS,
    }
    out = pil_img.resize(target, resample=_imap.get(interp, Image.BILINEAR))
    arr = np.array(out)

    # ---- 元dtype/レンジに戻す ----
    if is_float:
        if float_is_01:
            arr = (arr.astype(np.float32) / 255.0).astype(orig_dtype)  # 0..1で返す
        else:
            arr = arr.astype(orig_dtype)  # 0..255で返す
    else:
        arr = arr.astype(orig_dtype)
    return arr

# ========== 引数 ==========
parser = argparse.ArgumentParser()
parser.add_argument('--im_path', type=str, default='./demo/45765448.jpg',
                    help='input image path.')
parser.add_argument('--save_dir', type=str, default='outputs',
                    help='where to save results.')

# ========== カラーマップ ==========
floorplan_map = {
    0: [255,255,255], # background
    1: [192,192,224], # closet
    2: [192,255,255], # bathroom/washroom
    3: [224,255,192], # livingroom/kitchen/dining room
    4: [255,224,128], # bedroom
    5: [255,160, 96], # hall
    6: [255,224,224], # balcony
    7: [255,255,255], # not used
    8: [255,255,255], # not used
    9: [255, 60,128], # door & window
    10:[  0,  0,  0]  # wall
}

def ind2rgb(ind_im, color_map=floorplan_map):
    """ (H,W) のラベル → (H,W,3) uint8 """
    h, w = ind_im.shape[:2]
    rgb_im = np.zeros((h, w, 3), dtype=np.uint8)
    for i, rgb in color_map.items():
        rgb_im[(ind_im == i)] = rgb
    return rgb_im

def main(args):
    os.makedirs(args.save_dir, exist_ok=True)

    # 画像読込 & 前処理
    im = imread(args.im_path)  # imageio: カラーは(H,W,3)、グレーは(H,W)
    # ---- ★ここが追加：1ch/4chを3chに統一 ----
    if im.ndim == 2:
        # グレースケール → 3chへ複製
        im = np.stack([im, im, im], axis=-1)
    elif im.ndim == 3 and im.shape[2] == 4:
        # RGBA → 先頭3チャンネルのみ
        im = im[:, :, :3]

    im = im.astype(np.float32)
    im = imresize(im, (512, 512, 3)) / 255.0  # 0..1に正規化

    # 推論（TF1グラフをそのまま使用）
    with tf.Session() as sess:
        sess.run(tf.group(tf.global_variables_initializer(),
                          tf.local_variables_initializer()))

        saver = tf.train.import_meta_graph('./pretrained/pretrained_r3d.meta')
        saver.restore(sess, './pretrained/pretrained_r3d')

        graph = tf.get_default_graph()
        x = graph.get_tensor_by_name('inputs:0')
        room_type_logit = graph.get_tensor_by_name('Cast:0')
        room_boundary_logit = graph.get_tensor_by_name('Cast_1:0')

        room_type, room_boundary = sess.run(
            [room_type_logit, room_boundary_logit],
            feed_dict={x: im.reshape(1, 512, 512, 3)}
        )
        room_type = np.squeeze(room_type)
        room_boundary = np.squeeze(room_boundary)

    # マージ（部屋タイプ + エッジ上書き）
    floorplan = room_type.copy()
    floorplan[room_boundary == 1] = 9
    floorplan[room_boundary == 2] = 10
    floorplan_rgb = ind2rgb(floorplan)            # uint8
    im_uint8 = (np.clip(im, 0, 1) * 255).astype(np.uint8)

    # 保存（個別 & 横並び & Matplotlib 図）
    out_img = os.path.join(args.save_dir, 'input_512.png')
    out_seg = os.path.join(args.save_dir, 'floorplan_rgb.png')
    out_side = os.path.join(args.save_dir, 'vis_side_by_side.png')
    out_fig = os.path.join(args.save_dir, 'vis_matplotlib.png')

    imsave(out_img, im_uint8)
    imsave(out_seg, floorplan_rgb)
    side = np.concatenate([im_uint8, floorplan_rgb], axis=1)
    imsave(out_side, side)

    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1); plt.imshow(im_uint8);     plt.axis('off'); plt.title('Input')
    plt.subplot(1, 2, 2); plt.imshow(floorplan_rgb); plt.axis('off'); plt.title('Floorplan')
    plt.tight_layout()
    plt.savefig(out_fig, dpi=150)

    print(f"[Saved] {out_img}")
    print(f"[Saved] {out_seg}")
    print(f"[Saved] {out_side}")
    print(f"[Saved] {out_fig}")

if __name__ == '__main__':
    FLAGS, _ = parser.parse_known_args()
    main(FLAGS)
