# postprocess.py (Python 3-ready)
import argparse
import os
import sys
import glob

import numpy as np
from imageio.v2 import imread as imread_v2, imwrite as imsave_v2
from matplotlib import pyplot as plt  # 未使用でも元の依存を維持

# utils パスを通す（元コード踏襲）
sys.path.append('./utils/')
from rgb_ind_convertor import *  # ind2rgb, rgb2ind, floorplan_fuse_map
from util import *  # fill_break_line, flood_fill, refine_room_region 等

parser = argparse.ArgumentParser()
parser.add_argument(
    '--result_dir',
    type=str,
    default='./out',
    help='The folder that saves network predictions.'
)

def safe_imread(path):
    """
    画像をRGB uint8 (H, W, 3) に正規化して返す。
    - RGBA -> RGBに変換
    - グレースケール -> 3chに拡張
    """
    im = imread_v2(path)
    # dtype 正規化
    if np.issubdtype(im.dtype, np.floating):
        im = np.clip(im, 0.0, 1.0)
        im = (im * 255.0).astype(np.uint8)
    else:
        im = np.clip(im, 0, 255).astype(np.uint8)

    if im.ndim == 2:  # グレースケール -> 3ch
        im = np.stack([im, im, im], axis=-1)
    elif im.ndim == 3:
        # RGBA -> RGB
        if im.shape[-1] == 4:
            im = im[..., :3]
        elif im.shape[-1] == 3:
            pass
        else:
            # 想定外チャンネル数は最初の3chのみ使用
            im = im[..., :3]
    else:
        raise ValueError(f"Unsupported image shape {im.shape} for {path}")
    return im

def imsave(path, arr):
    """imageio で保存（uint8, RGB想定）。"""
    if arr.dtype != np.uint8:
        arr = np.clip(arr, 0, 255).astype(np.uint8)
    # RGBA の可能性があれば 3ch に落とす
    if arr.ndim == 3 and arr.shape[-1] > 3:
        arr = arr[..., :3]
    imsave_v2(path, arr)

def post_process(input_dir, save_dir, merge=True):
    os.makedirs(save_dir, exist_ok=True)

    input_paths = sorted(glob.glob(os.path.join(input_dir, '*.png')))
    names = [os.path.basename(i) for i in input_paths]
    out_paths = [os.path.join(save_dir, i) for i in names]

    n = len(input_paths)
    for i in range(n):  # xrange -> range
        im = safe_imread(input_paths[i])
        im_ind = rgb2ind(im, color_map=floorplan_fuse_map)

        # room と boundary を分離
        rm_ind = im_ind.copy()
        rm_ind[im_ind == 9] = 0
        rm_ind[im_ind == 10] = 0

        bd_ind = np.zeros(im_ind.shape, dtype=np.uint8)
        bd_ind[im_ind == 9] = 9
        bd_ind[im_ind == 10] = 10

        hard_c = (bd_ind > 0).astype(np.uint8)

        # room 自身の領域
        rm_mask = np.zeros(rm_ind.shape, dtype=np.uint8)
        rm_mask[rm_ind > 0] = 1

        # close wall 由来の領域（明線の切れを補修）
        cw_mask = fill_break_line(hard_c)

        fuse_mask = cw_mask + rm_mask
        fuse_mask[fuse_mask >= 1] = 255

        # 穴埋め
        fuse_mask = flood_fill(fuse_mask)
        fuse_mask = fuse_mask // 255

        # 1部屋1ラベル化
        new_rm_ind = refine_room_region(cw_mask, rm_ind)

        # 背景の誤分類を無視
        new_rm_ind = fuse_mask * new_rm_ind

        print(f"Saving {i}th refined room prediction to {out_paths[i]}")

        if merge:
            # 境界ラベルを戻す
            new_rm_ind[bd_ind == 9] = 9
            new_rm_ind[bd_ind == 10] = 10
            rgb = ind2rgb(new_rm_ind, color_map=floorplan_fuse_map)
        else:
            rgb = ind2rgb(new_rm_ind)

        imsave(out_paths[i], rgb)

if __name__ == '__main__':
    FLAGS, _ = parser.parse_known_args()
    input_dir = FLAGS.result_dir
    save_dir = os.path.join(input_dir, 'post')
    post_process(input_dir, save_dir)
