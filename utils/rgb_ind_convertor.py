# -*- coding: utf-8 -*-
"""
utils/rgb_ind_convertor.py — Python3 & SciPy非依存 互換版
- dict.iteritems() → dict.items()
- scipy.misc.toimage → Pillow(Image.fromarray)
- 画素比較のロバスト化（float→uint8 換算、(H,W,1) 対応 など）
"""

import numpy as np
from PIL import Image

# ========== Color Maps (original values preserved) ==========
# use for index 2 rgb
floorplan_room_map = {
    0: [  0,  0,  0], # background
    1: [192,192,224], # closet
    2: [192,255,255], # bathroom/washroom
    3: [224,255,192], # livingroom/kitchen/diningroom
    4: [255,224,128], # bedroom
    5: [255,160, 96], # hall
    6: [255,224,224], # balcony
    7: [224,224,224], # not used
    8: [224,224,128]  # not used
}

# boundary label
floorplan_boundary_map = {
    0: [  0,  0,  0], # background
    1: [255, 60,128], # opening (door&window)
    2: [255,255,255]  # wall line
}

# boundary label for presentation
floorplan_boundary_map_figure = {
    0: [255,255,255], # background
    1: [255, 60,128], # opening (door&window)
    2: [  0,  0,  0]  # wall line
}

# merge all label into one multi-class label
floorplan_fuse_map = {
    0:  [  0,  0,  0], # background
    1:  [192,192,224], # closet
    2:  [192,255,255], # bathroom/washroom
    3:  [224,255,192], # livingroom/kitchen/dining room
    4:  [255,224,128], # bedroom
    5:  [255,160, 96], # hall
    6:  [255,224,224], # balcony
    7:  [224,224,224], # not used
    8:  [224,224,128], # not used
    9:  [255, 60,128], # extra label for opening (door&window)
    10: [255,255,255]  # extra label for wall line
}

# invert the color of wall line and background for presentation
floorplan_fuse_map_figure = {
    0:  [255,255,255], # background
    1:  [192,192,224], # closet
    2:  [192,255,255], # bathroom/washroom
    3:  [224,255,192], # livingroom/kitchen/dining room
    4:  [255,224,128], # bedroom
    5:  [255,160, 96], # hall
    6:  [255,224,224], # balcony
    7:  [224,224,224], # not used
    8:  [224,224,128], # not used
    9:  [255, 60,128], # extra label for opening (door&window)
    10: [  0,  0,  0]  # extra label for wall line
}

# ========== Helpers ==========

def _to_uint8_image(arr):
    """
    任意の画像配列を 0..255 の uint8 RGB/グレースケールに正規化。
    (H,W), (H,W,1), (H,W,3) を想定。float は [0,1] または [0,255] を想定。
    """
    a = np.asarray(arr)
    if a.ndim == 3 and a.shape[-1] == 1:
        a = a[..., 0]

    # 型とレンジを整える
    if np.issubdtype(a.dtype, np.floating):
        # 推定: 0..1 or 0..255
        maxv = float(np.nanmax(a)) if a.size else 1.0
        scale = 255.0 if maxv <= 1.0 + 1e-6 else 1.0
        a = np.clip(a * scale, 0.0, 255.0).astype(np.uint8)
    else:
        a = np.clip(a, 0, 255).astype(np.uint8)
    return a

def _ensure_rgb(arr):
    """
    (H,W) or (H,W,1) → (H,W,3) に拡張。既にRGBなら先頭3チャネルに統一。
    """
    a = np.asarray(arr)
    if a.ndim == 2:
        a = np.stack([a, a, a], axis=-1)
    elif a.ndim == 3:
        if a.shape[-1] == 1:
            a = np.concatenate([a, a, a], axis=-1)
        elif a.shape[-1] >= 3:
            a = a[..., :3]
        else:
            raise ValueError(f"Unsupported channel number: {a.shape}")
    else:
        raise ValueError(f"Unsupported rank: {a.ndim}")
    return a

# ========== Public APIs ==========

def rgb2ind(im, color_map=floorplan_room_map):
    """
    RGB画像 → クラスインデックス（uint8）
    im: (H,W,3) or (H,W,1) or (H,W) ; 値は uint8 か float
    color_map: {index: [R,G,B]}
    """
    img = _ensure_rgb(_to_uint8_image(im))
    h, w, _ = img.shape
    ind = np.zeros((h, w), dtype=np.uint8)

    # Python3: items()
    for cls, rgb in color_map.items():
        # color_map のキーは int 前提。文字列キーなら明示変換。
        try:
            k = int(cls)
        except Exception:
            k = cls
        rgb = np.array(rgb, dtype=np.uint8)
        mask = (img == rgb).all(axis=2)
        ind[mask] = np.uint8(k)
    return ind

def ind2rgb(ind_im, color_map=floorplan_room_map):
    """
    クラスインデックス（2D/3D）→ RGB 画像（uint8）
    ind_im: (H,W) or (H,W,1) ; 値は int
    color_map: {index: [R,G,B]}
    """
    ind = np.asarray(ind_im)
    if ind.ndim == 3 and ind.shape[-1] == 1:
        ind = ind[..., 0]
    if not np.issubdtype(ind.dtype, np.integer):
        ind = ind.astype(np.int32, copy=False)

    h, w = ind.shape[:2]
    rgb_im = np.zeros((h, w, 3), dtype=np.uint8)

    # Python3: items()
    for cls, rgb in color_map.items():
        try:
            k = int(cls)
        except Exception:
            k = cls
        mask = (ind == k)
        if np.any(mask):
            rgb_im[mask] = np.array(rgb, dtype=np.uint8)

    return rgb_im

def unscale_imsave(path, im, cmin=0, cmax=255):
    """
    旧 scipy.misc.toimage(...).save(path) の代替。
    - im を 0..255 uint8 に整えて保存
    - グレースケール/カラーどちらもOK
    """
    a = np.asarray(im)
    # cmin/cmax を尊重して 0..255 に線形スケーリング
    a = np.clip(a, cmin, cmax)
    if cmax > cmin:
        a = (a - cmin) * (255.0 / float(cmax - cmin))
    a = _to_uint8_image(a)
    a = _ensure_rgb(a)  # RGBに統一（元がLでもOK）
    Image.fromarray(a).save(path)
