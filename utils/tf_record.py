# -*- coding: utf-8 -*-
"""
tf_record.py — SciPy依存除去 & Python3/TF2(compat.v1) 互換版
- scipy.misc の imread / imresize / imsave を廃止
  -> imageio.v2.imread / imageio.v2.imwrite + Pillow で imresize を再実装
- TF1 キュー/TFRecord API は tf.compat.v1.* に統一
- バイナリ書き込みは .tobytes() を使用
"""

from __future__ import print_function
import os, sys, glob, time
import numpy as np
import tensorflow as tf

from imageio.v2 import imread as _imread, imwrite as _imsave
from PIL import Image

# このファイルは utils/ 配下。呼び出し側(net.py)で sys.path に ./utils を追加済み
from rgb_ind_convertor import *  # ind2rgb, rgb2ind, color maps 等

# ---- TF1 API を使う前提（main.py側で disable_eager_execution 済み） ----
# ここでは明示的に compat.v1 を使う
v1 = tf.compat.v1

# ========== 互換 imresize 実装 ==========
def imresize(img, size, interp='bilinear', mode=None):
    """
    PILベースの imresize 互換:
      - size: (H, W) or (H, W, C)
      - dtype/range は入力を踏襲（uint8はそのまま、floatは[0,1]想定で255スケール→戻す）
    """
    # size 正規化
    if isinstance(size, (list, tuple)):
        if len(size) == 2:
            out_hw = (int(size[0]), int(size[1]))
            want_c = None
        elif len(size) == 3:
            out_hw = (int(size[0]), int(size[1]))
            want_c = int(size[2])
        else:
            raise ValueError("size must be (H,W) or (H,W,C)")
    else:
        raise ValueError("size must be (H,W) or (H,W,C)")

    orig_dtype = img.dtype
    is_float = np.issubdtype(orig_dtype, np.floating)

    # 入力→PIL 変換（uint8 RGB/L）
    if is_float:
        # [0,1] 想定で 0-255 に
        src = np.clip(img, 0.0, 1.0) * 255.0
        src = src.astype(np.uint8)
    else:
        src = img
        if src.dtype != np.uint8:
            src = np.clip(src, 0, 255).astype(np.uint8)

    # グレースケール or RGB 判定
    if src.ndim == 2:
        pil = Image.fromarray(src, mode='L')
    elif src.ndim == 3:
        if src.shape[2] == 1:
            pil = Image.fromarray(src.squeeze(-1), mode='L')
        elif src.shape[2] >= 3:
            pil = Image.fromarray(src[..., :3], mode='RGB')
        else:
            raise ValueError("Unsupported channel size: {}".format(src.shape))
    else:
        raise ValueError("Unsupported rank: {}".format(src.ndim))

    # 補間法
    interp_map = {
        'nearest': Image.NEAREST,
        'bilinear': Image.BILINEAR,
        'bicubic': Image.BICUBIC,
        'lanczos': Image.LANCZOS,
        None: Image.BILINEAR,
    }
    resample = interp_map.get(interp, Image.BILINEAR)

    out = pil.resize((out_hw[1], out_hw[0]), resample=resample)  # PILは(W,H)

    # PIL→ndarrayへ戻す
    out_np = np.array(out)

    # チャンネル合わせ（必要なら3チャンネル化）
    if want_c is not None:
        if want_c == 3:
            if out_np.ndim == 2:
                out_np = np.stack([out_np]*3, axis=-1)
            elif out_np.ndim == 3 and out_np.shape[2] == 1:
                out_np = np.concatenate([out_np]*3, axis=-1)
            elif out_np.ndim == 3 and out_np.shape[2] >= 3:
                out_np = out_np[..., :3]
        elif want_c == 1:
            if out_np.ndim == 3:
                # RGB→L
                out_np = np.array(Image.fromarray(out_np).convert('L'))
        # 他は out_np をそのまま

    # dtype/range を入力に合わせて戻す
    if is_float:
        out_np = out_np.astype(np.float32) / 255.0
    else:
        out_np = out_np.astype(orig_dtype)

    # (H,W) 指定時は rank を入力に合わせる
    if isinstance(size, (list, tuple)) and len(size) == 2:
        # 入力が3chで(H,W)指定なら、出力も2Dで返す（呼び出し側で整形する想定）
        pass

    return out_np

# シンプルなimread/imsaveのラッパ
def imread(path, mode=None):
    """
    mode: 'RGB' or 'L' を想定（Noneなら原画像）
    """
    if mode is None:
        img = _imread(path)
    elif mode.upper() == 'RGB':
        img = _imread(path, pilmode='RGB')
    elif mode.upper() == 'L':
        img = _imread(path, pilmode='L')
    else:
        img = _imread(path)
    return img

def imsave(path, array):
    _imsave(path, array)

# ===== TFRecord ユーティリティ =====
def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[int(value)]))

def _bytes_feature(value: bytes):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

# ------------------------------------------------------------
# Raw → TFRecord（学習: 4ラベル）
def load_raw_images(path_line):
    paths = path_line.split('\t')

    image = imread(paths[0], mode='RGB')
    wall  = imread(paths[1], mode='L')
    close = imread(paths[2], mode='L')
    room  = imread(paths[3], mode='RGB')
    close_wall = imread(paths[4], mode='L')

    image = imresize(image, (512, 512, 3))
    wall = imresize(wall, (512, 512))
    close = imresize(close, (512, 512))
    close_wall = imresize(close_wall, (512, 512))
    room = imresize(room, (512, 512, 3))

    room_ind = rgb2ind(room)

    image = image.astype(np.uint8)
    wall = wall.astype(np.uint8)
    close = close.astype(np.uint8)
    close_wall = close_wall.astype(np.uint8)
    room_ind = room_ind.astype(np.uint8)

    return image, wall, close, room_ind, close_wall

def write_record(paths, name='dataset.tfrecords'):
    with tf.io.TFRecordWriter(name) as writer:
        for i in range(len(paths)):
            image, wall, close, room_ind, close_wall = load_raw_images(paths[i])
            feature = {
                'image': _bytes_feature(image.tobytes()),
                'wall': _bytes_feature(wall.tobytes()),
                'close': _bytes_feature(close.tobytes()),
                'room': _bytes_feature(room_ind.tobytes()),
                'close_wall': _bytes_feature(close_wall.tobytes()),
            }
            example = tf.train.Example(features=tf.train.Features(feature=feature))
            writer.write(example.SerializeToString())

def read_record(data_path, batch_size=1, size=512):
    feature = {
        'image': v1.FixedLenFeature(shape=(), dtype=tf.string),
        'wall': v1.FixedLenFeature(shape=(), dtype=tf.string),
        'close': v1.FixedLenFeature(shape=(), dtype=tf.string),
        'room': v1.FixedLenFeature(shape=(), dtype=tf.string),
        'close_wall': v1.FixedLenFeature(shape=(), dtype=tf.string),
    }

    filename_queue = v1.train.string_input_producer(
        [data_path], num_epochs=None, shuffle=False, capacity=batch_size*128)

    reader = v1.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)

    features = v1.parse_single_example(serialized_example, features=feature)

    image = v1.decode_raw(features['image'], tf.uint8)
    wall = v1.decode_raw(features['wall'], tf.uint8)
    close = v1.decode_raw(features['close'], tf.uint8)
    room = v1.decode_raw(features['room'], tf.uint8)
    close_wall = v1.decode_raw(features['close_wall'], tf.uint8)

    image = tf.cast(image, dtype=tf.float32)
    wall = tf.cast(wall, dtype=tf.float32)
    close = tf.cast(close, dtype=tf.float32)
    close_wall = tf.cast(close_wall, dtype=tf.float32)

    image = v1.reshape(image, [size, size, 3])
    wall = v1.reshape(wall, [size, size, 1])
    close = v1.reshape(close, [size, size, 1])
    room = v1.reshape(room, [size, size])
    close_wall = v1.reshape(close_wall, [size, size, 1])

    image = image / 255.0
    wall = wall / 255.0
    close = close / 255.0
    close_wall = close_wall / 255.0

    room_one_hot = tf.one_hot(tf.cast(room, tf.int32), 9, axis=-1)

    images, walls, closes, rooms, close_walls = v1.train.shuffle_batch(
        [image, wall, close, room_one_hot, close_wall],
        batch_size=batch_size, capacity=batch_size*128,
        num_threads=1, min_after_dequeue=batch_size*32)

    return {'images': images, 'walls': walls, 'closes': closes,
            'rooms': rooms, 'close_walls': close_walls}

# ------------------------------------------------------------
# セグメンテーション（統合ラベル）用
def load_seg_raw_images(path_line):
    paths = path_line.split('\t')

    image = imread(paths[0], mode='RGB')
    close = imread(paths[2], mode='L')
    room  = imread(paths[3], mode='RGB')
    close_wall = imread(paths[4], mode='L')

    image = imresize(image, (512, 512, 3))
    close = imresize(close, (512, 512)) / 255.0
    close_wall = imresize(close_wall, (512, 512)) / 255.0
    room = imresize(room, (512, 512, 3))

    room_ind = rgb2ind(room)

    d_ind = (close > 0.5).astype(np.uint8)
    cw_ind = (close_wall > 0.5).astype(np.uint8)

    room_ind[cw_ind == 1] = 10
    room_ind[d_ind == 1] = 9

    image = image.astype(np.uint8)
    room_ind = room_ind.astype(np.uint8)
    return image, room_ind

def write_seg_record(paths, name='dataset.tfrecords'):
    with tf.io.TFRecordWriter(name) as writer:
        for i in range(len(paths)):
            image, room_ind = load_seg_raw_images(paths[i])
            feature = {
                'image': _bytes_feature(image.tobytes()),
                'label': _bytes_feature(room_ind.tobytes()),
            }
            example = tf.train.Example(features=tf.train.Features(feature=feature))
            writer.write(example.SerializeToString())

def read_seg_record(data_path, batch_size=1, size=512):
    feature = {
        'image': v1.FixedLenFeature(shape=(), dtype=tf.string),
        'label': v1.FixedLenFeature(shape=(), dtype=tf.string),
    }

    filename_queue = v1.train.string_input_producer(
        [data_path], num_epochs=None, shuffle=False, capacity=batch_size*128)

    reader = v1.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)

    features = v1.parse_single_example(serialized_example, features=feature)

    image = v1.decode_raw(features['image'], tf.uint8)
    label = v1.decode_raw(features['label'], tf.uint8)

    image = tf.cast(image, dtype=tf.float32)

    image = v1.reshape(image, [size, size, 3])
    label = v1.reshape(label, [size, size])

    image = image / 255.0
    label_one_hot = tf.one_hot(tf.cast(label, tf.int32), 11, axis=-1)

    images, labels = v1.train.shuffle_batch(
        [image, label_one_hot],
        batch_size=batch_size, capacity=batch_size*128,
        num_threads=1, min_after_dequeue=batch_size*32)

    return {'images': images, 'labels': labels}

# ------------------------------------------------------------
# マルチタスク（boundary と room）
def load_bd_rm_images(path_line):
    paths = path_line.split('\t')

    image = imread(paths[0], mode='RGB')
    close = imread(paths[2], mode='L')
    room  = imread(paths[3], mode='RGB')
    close_wall = imread(paths[4], mode='L')

    image = imresize(image, (512, 512, 3))
    close = imresize(close, (512, 512)) / 255.0
    close_wall = imresize(close_wall, (512, 512)) / 255.0
    room = imresize(room, (512, 512, 3))

    room_ind = rgb2ind(room)

    d_ind = (close > 0.5).astype(np.uint8)
    cw_ind = (close_wall > 0.5).astype(np.uint8)

    cw_ind[cw_ind == 1] = 2
    cw_ind[d_ind == 1] = 1

    image = image.astype(np.uint8)
    room_ind = room_ind.astype(np.uint8)
    cw_ind = cw_ind.astype(np.uint8)

    return image, cw_ind, room_ind, d_ind

def write_bd_rm_record(paths, name='dataset.tfrecords'):
    with tf.io.TFRecordWriter(name) as writer:
        for i in range(len(paths)):
            image, cw_ind, room_ind, d_ind = load_bd_rm_images(paths[i])
            feature = {
                'image': _bytes_feature(image.tobytes()),
                'boundary': _bytes_feature(cw_ind.tobytes()),
                'room': _bytes_feature(room_ind.tobytes()),
                'door': _bytes_feature(d_ind.tobytes()),
            }
            example = tf.train.Example(features=tf.train.Features(feature=feature))
            writer.write(example.SerializeToString())

def read_bd_rm_record(data_path, batch_size=1, size=512):
    feature = {
        'image': v1.FixedLenFeature(shape=(), dtype=tf.string),
        'boundary': v1.FixedLenFeature(shape=(), dtype=tf.string),
        'room': v1.FixedLenFeature(shape=(), dtype=tf.string),
        'door': v1.FixedLenFeature(shape=(), dtype=tf.string),
    }

    filename_queue = v1.train.string_input_producer(
        [data_path], num_epochs=None, shuffle=False, capacity=batch_size*128)

    reader = v1.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)

    features = v1.parse_single_example(serialized_example, features=feature)

    image = v1.decode_raw(features['image'], tf.uint8)
    boundary = v1.decode_raw(features['boundary'], tf.uint8)
    room = v1.decode_raw(features['room'], tf.uint8)
    door = v1.decode_raw(features['door'], tf.uint8)

    image = tf.cast(image, dtype=tf.float32)

    image = v1.reshape(image, [size, size, 3])
    boundary = v1.reshape(boundary, [size, size])
    room = v1.reshape(room, [size, size])
    door = v1.reshape(door, [size, size])

    image = image / 255.0

    label_boundary = tf.one_hot(tf.cast(boundary, tf.int32), 3, axis=-1)
    label_room = tf.one_hot(tf.cast(room, tf.int32), 9, axis=-1)

    images, label_boundaries, label_rooms, label_doors = v1.train.shuffle_batch(
        [image, label_boundary, label_room, door],
        batch_size=batch_size, capacity=batch_size*128,
        num_threads=1, min_after_dequeue=batch_size*32)

    return {'images': images,
            'label_boundaries': label_boundaries,
            'label_rooms': label_rooms,
            'label_doors': label_doors}
