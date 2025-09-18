# -*- coding: utf-8 -*-
"""
DeepFloorplan net.py — Python3 / TF2互換(compat.v1) 版
- print文 → print()
- tf.* を tf.compat.v1.* で統一（variable_scope, get_variable, global_variables 等）
- tf.image.resize_images → tf.image.resize
- tf.nn.moments の keep_dims → keepdims
- SciPy misc の imread/imsave/imresize 依存を除去（本ファイルでは未使用）
"""

from __future__ import print_function
import os
import sys
import time
import random
import glob
import numpy as np
import tensorflow as tf

# TF2 環境で TF1 API を使う
tf.compat.v1.disable_eager_execution()

# （参考）古い contrib.vgg の import はTF2で廃止。pretrained復元は init_from_checkpoint を直接利用するため必須ではない。
try:
    from tensorflow.contrib.slim.nets import vgg  # noqa: F401
except Exception:
    vgg = None

# utils パス
sys.path.append('./utils/')
from rgb_ind_convertor import *          # ind2rgb, rgb2ind, color maps
from util import fast_hist               # 評価用
from tf_record import read_record, read_bd_rm_record

GPU_ID = '0'

def data_loader_bd_rm_from_tfrecord(batch_size=1):
    # 元実装は '../dataset' を見にいく
    paths = open('../dataset/r3d_train.txt', 'r').read().splitlines()
    loader_dict = read_bd_rm_record('../dataset/r3d.tfrecords', batch_size=batch_size, size=512)
    num_batch = len(paths) // batch_size
    return loader_dict, num_batch


class Network(object):
    """Backbone + デコーダ"""
    def __init__(self, dtype=tf.float32):
        print('Initial nn network object...')
        self.dtype = dtype
        # VGG16 事前学習の復元マップ（必要時のみ使用）
        self.pre_train_restore_map = {
            'vgg_16/conv1/conv1_1/weights':'FNet/conv1_1/W',
            'vgg_16/conv1/conv1_1/biases':'FNet/conv1_1/b',
            'vgg_16/conv1/conv1_2/weights':'FNet/conv1_2/W',
            'vgg_16/conv1/conv1_2/biases':'FNet/conv1_2/b',
            'vgg_16/conv2/conv2_1/weights':'FNet/conv2_1/W',
            'vgg_16/conv2/conv2_1/biases':'FNet/conv2_1/b',
            'vgg_16/conv2/conv2_2/weights':'FNet/conv2_2/W',
            'vgg_16/conv2/conv2_2/biases':'FNet/conv2_2/b',
            'vgg_16/conv3/conv3_1/weights':'FNet/conv3_1/W',
            'vgg_16/conv3/conv3_1/biases':'FNet/conv3_1/b',
            'vgg_16/conv3/conv3_2/weights':'FNet/conv3_2/W',
            'vgg_16/conv3/conv3_2/biases':'FNet/conv3_2/b',
            'vgg_16/conv3/conv3_3/weights':'FNet/conv3_3/W',
            'vgg_16/conv3/conv3_3/biases':'FNet/conv3_3/b',
            'vgg_16/conv4/conv4_1/weights':'FNet/conv4_1/W',
            'vgg_16/conv4/conv4_1/biases':'FNet/conv4_1/b',
            'vgg_16/conv4/conv4_2/weights':'FNet/conv4_2/W',
            'vgg_16/conv4/conv4_2/biases':'FNet/conv4_2/b',
            'vgg_16/conv4/conv4_3/weights':'FNet/conv4_3/W',
            'vgg_16/conv4/conv4_3/biases':'FNet/conv4_3/b',
            'vgg_16/conv5/conv5_1/weights':'FNet/conv5_1/W',
            'vgg_16/conv5/conv5_1/biases':'FNet/conv5_1/b',
            'vgg_16/conv5/conv5_2/weights':'FNet/conv5_2/W',
            'vgg_16/conv5/conv5_2/biases':'FNet/conv5_2/b',
            'vgg_16/conv5/conv5_3/weights':'FNet/conv5_3/W',
            'vgg_16/conv5/conv5_3/biases':'FNet/conv5_3/b'
        }

    # ===== 基本レイヤ =====
    def _he_uniform(self, shape, regularizer=None, trainable=None, name=None):
        name = 'W' if name is None else name + '/W'
        kernel_size = np.prod(shape[:2])     # k_h*k_w
        fan_in = shape[-2] * kernel_size
        s = np.sqrt(1. / max(1.0, fan_in))
        with tf.device('/device:GPU:' + GPU_ID):
            w = tf.compat.v1.get_variable(
                name, shape, dtype=self.dtype,
                initializer=tf.compat.v1.random_uniform_initializer(minval=-s, maxval=s),
                regularizer=regularizer, trainable=trainable)
        return w

    def _constant(self, shape, value=0, regularizer=None, trainable=None, name=None):
        name = 'b' if name is None else name + '/b'
        with tf.device('/device:GPU:' + GPU_ID):
            b = tf.compat.v1.get_variable(
                name, shape, dtype=self.dtype,
                initializer=tf.compat.v1.constant_initializer(value=value),
                regularizer=regularizer, trainable=trainable)
        return b

    def _conv2d(self, tensor, dim, size=3, stride=1, rate=1, pad='SAME',
                act='relu', norm='none', G=16, bias=True, name='conv'):
        """(act → norm) → conv"""
        in_dim = tensor.shape.as_list()[-1]
        size = size if isinstance(size, (tuple, list)) else [size, size]
        stride = stride if isinstance(stride, (tuple, list)) else [1, stride, stride, 1]
        rate = rate if isinstance(rate, (tuple, list)) else [1, rate, rate, 1]
        kernel_shape = [size[0], size[1], in_dim, dim]

        w = self._he_uniform(kernel_shape, name=name)
        b = self._constant(dim, name=name) if bias else 0

        # activation
        if act == 'relu':
            tensor = tf.nn.relu(tensor, name=name + '/relu')
        elif act == 'sigmoid':
            tensor = tf.nn.sigmoid(tensor, name=name + '/sigmoid')
        elif act == 'softplus':
            tensor = tf.nn.softplus(tensor, name=name + '/softplus')
        elif act == 'leaky_relu':
            tensor = tf.nn.leaky_relu(tensor, name=name + '/leaky_relu')
        else:
            norm = 'none'

        # group norm（activation 後）
        if norm == 'gn':
            x = tf.transpose(tensor, [0, 3, 1, 2])  # NHWC→NCHW
            N, C, H, W = x.get_shape().as_list()
            G = min(G, C)
            x = tf.reshape(x, [-1, G, C // G, H, W])
            mean, var = tf.nn.moments(x, [2, 3, 4], keepdims=True)
            x = (x - mean) / tf.sqrt(var + 1e-6)
            with tf.device('/device:GPU:' + GPU_ID):
                gamma = tf.compat.v1.get_variable(name + '/gamma', [C], dtype=self.dtype,
                                     initializer=tf.compat.v1.constant_initializer(1.0))
                beta  = tf.compat.v1.get_variable(name + '/beta',  [C], dtype=self.dtype,
                                     initializer=tf.compat.v1.constant_initializer(0.0))
                gamma = tf.reshape(gamma, [1, C, 1, 1])
                beta  = tf.reshape(beta,  [1, C, 1, 1])
            tensor = tf.reshape(x, [-1, C, H, W]) * gamma + beta
            tensor = tf.transpose(tensor, [0, 2, 3, 1])  # NCHW→NHWC

        out = tf.nn.conv2d(tensor, w, strides=stride, padding=pad, dilations=rate, name=name)
        if bias:
            out = out + b
        return out

    def _upconv2d(self, tensor, dim, size=4, stride=2, pad='SAME', act='relu', name='upconv'):
        # NHWC
        shape_list = tensor.shape.as_list()
        batch_size, h, w, in_dim = shape_list
        size = size if isinstance(size, (tuple, list)) else [size, size]
        stride = stride if isinstance(stride, (tuple, list)) else [1, stride, stride, 1]
        kernel_shape = [size[0], size[1], dim, in_dim]
        W = self._he_uniform(kernel_shape, name=name)

        if pad == 'SAME':
            out_h = h * stride[1]
            out_w = w * stride[2]
        else:
            out_h = (h - 1) * stride[1] + size[0]
            out_w = (w - 1) * stride[2] + size[1]

        # 動的バッチ対応（1 など固定ならそのままでもOK）
        dyn_bs = tf.shape(tensor)[0] if batch_size is None else batch_size
        out_shape = tf.stack([dyn_bs, out_h, out_w, dim])

        out = tf.nn.conv2d_transpose(tensor, W, output_shape=out_shape,
                                     strides=stride, padding=pad, name=name)
        # 静的 shape をできる範囲で設定
        out.set_shape([batch_size, out_h, out_w, dim])

        if act == 'relu':
            out = tf.nn.relu(out, name=name + '/relu')
        elif act == 'sigmoid':
            out = tf.nn.sigmoid(out, name=name + '/sigmoid')
        return out

    def _max_pool2d(self, tensor, size=2, stride=2, pad='VALID'):
        size = size if isinstance(size, (tuple, list)) else [1, size, size, 1]
        stride = stride if isinstance(stride, (tuple, list)) else [1, stride, stride, 1]
        size = [1, size[0], size[1], 1] if len(size) == 2 else size
        stride = [1, stride[0], stride[1], 1] if len(stride) == 2 else stride
        out = tf.nn.max_pool(tensor, ksize=size, strides=stride, padding=pad)
        return out

    # ===== コンテキスト結合系 =====
    def _constant_kernel(self, shape, value=1.0, diag=False, flip=False,
                         regularizer=None, trainable=None, name=None):
        name = 'fixed_w' if name is None else name + '/fixed_w'
        with tf.device('/device:GPU:' + GPU_ID):
            if not diag:
                k = tf.compat.v1.get_variable(
                    name, shape, dtype=self.dtype,
                    initializer=tf.compat.v1.constant_initializer(value=value),
                    regularizer=regularizer, trainable=trainable)
            else:
                w = tf.eye(shape[0], num_columns=shape[1])
                if flip:
                    w = tf.reshape(w, (shape[0], shape[1], 1))
                    w = tf.image.flip_left_right(w)
                w = tf.reshape(w, shape)
                k = tf.compat.v1.get_variable(
                    name, dtype=self.dtype, initializer=w,
                    regularizer=regularizer, trainable=trainable)
        return k

    def _context_conv2d(self, tensor, dim=1, size=7, diag=False, flip=False, stride=1, name='cconv'):
        in_dim = tensor.shape.as_list()[-1]  # 想定:1
        size = size if isinstance(size, (tuple, list)) else [size, size]
        stride = stride if isinstance(stride, (tuple, list)) else [1, stride, stride, 1]
        kernel_shape = [size[0], size[1], in_dim, dim]
        w = self._constant_kernel(kernel_shape, diag=diag, flip=flip, trainable=False, name=name)
        out = tf.nn.conv2d(tensor, w, strides=stride, padding='SAME', name=name)
        return out

    def _non_local_context(self, tensor1, tensor2, stride=4, name='non_local_context'):
        """Self-Attention 的に tensor1 を鍵に tensor2 を強調、さらに水平/垂直/斜めのコンテキスト合成"""
        assert tensor1.shape.as_list() == tensor2.shape.as_list(), "input tensor should have same shape"
        N, H, W, C = tensor1.shape.as_list()

        hs = H // stride if (H // stride) > 1 else (stride - 1)
        vs = W // stride if (W // stride) > 1 else (stride - 1)
        hs = hs if (hs % 2 != 0) else hs + 1
        vs = vs if (vs % 2 != 0) else vs + 1

        a = self._conv2d(tensor1, dim=C, name=name + '/fa1')
        a = self._conv2d(a, dim=C, name=name + '/fa2')
        a = self._conv2d(a, dim=1, size=1, act='linear', norm=None, name=name + '/a')
        a = tf.nn.sigmoid(a, name=name + '/a_sigmoid')

        x = self._conv2d(tensor2, dim=C, name=name + '/fx1')
        x = self._conv2d(x, dim=1, size=1, act='linear', norm=None, name=name + '/x')

        x = a * x

        h  = self._context_conv2d(x, size=[hs, 1], name=name + '/cc_h')
        v  = self._context_conv2d(x, size=[1, vs], name=name + '/cc_v')
        d1 = self._context_conv2d(x, size=[hs, vs], diag=True, name=name + '/cc_d1')
        d2 = self._context_conv2d(x, size=[hs, vs], diag=True, flip=True, name=name + '/cc_d2')

        c1 = a * (h + v + d1 + d2)
        c1 = self._conv2d(c1, dim=C, size=1, act='linear', norm=None, name=name + '/expand')

        features = tf.concat([tensor2, c1], axis=3, name=name + '/in_context_concat')
        out = self._conv2d(features, dim=C, name=name + '/conv2')
        return out, None

    def _up_bilinear(self, tensor, dim, shape, name='upsample'):
        out = self._conv2d(tensor, dim=dim, size=1, act='linear', name=name + '/1x1_conv')
        # tf.image.resize_images は非推奨 → tf.image.resize
        out = tf.image.resize(out, shape, method='bilinear')
        return out

    # ===== ネット本体 =====
    def forward(self, inputs, init_with_pretrain_vgg=False, pre_trained_model='./vgg16/vgg_16.ckpt'):
        # 共有バックボーン
        reuse_fnet = len([v for v in tf.compat.v1.global_variables() if v.name.startswith('FNet')]) > 0
        with tf.compat.v1.variable_scope('FNet', reuse=reuse_fnet):
            self.conv1_1 = self._conv2d(inputs, dim=64, name='conv1_1')
            self.conv1_2 = self._conv2d(self.conv1_1, dim=64, name='conv1_2')
            self.pool1   = self._max_pool2d(self.conv1_2)   # /2

            self.conv2_1 = self._conv2d(self.pool1, dim=128, name='conv2_1')
            self.conv2_2 = self._conv2d(self.conv2_1, dim=128, name='conv2_2')
            self.pool2   = self._max_pool2d(self.conv2_2)   # /4

            self.conv3_1 = self._conv2d(self.pool2, dim=256, name='conv3_1')
            self.conv3_2 = self._conv2d(self.conv3_1, dim=256, name='conv3_2')
            self.conv3_3 = self._conv2d(self.conv3_2, dim=256, name='conv3_3')
            self.pool3   = self._max_pool2d(self.conv3_3)   # /8

            self.conv4_1 = self._conv2d(self.pool3, dim=512, name='conv4_1')
            self.conv4_2 = self._conv2d(self.conv4_1, dim=512, name='conv4_2')
            self.conv4_3 = self._conv2d(self.conv4_2, dim=512, name='conv4_3')
            self.pool4   = self._max_pool2d(self.conv4_3)   # /16

            self.conv5_1 = self._conv2d(self.pool4, dim=512, name='conv5_1')
            self.conv5_2 = self._conv2d(self.conv5_1, dim=512, name='conv5_2')
            self.conv5_3 = self._conv2d(self.conv5_2, dim=512, name='conv5_3')
            self.pool5   = self._max_pool2d(self.conv5_3)   # /32

            if init_with_pretrain_vgg:
                tf.compat.v1.train.init_from_checkpoint(pre_trained_model, self.pre_train_restore_map)

            n, h, w, c = inputs.shape.as_list()

        # 近傍壁（CW）ブランチ
        reuse_cw_net = len([v for v in tf.compat.v1.global_variables() if v.name.startswith('CWNet')]) > 0
        with tf.compat.v1.variable_scope('CWNet', reuse=reuse_cw_net):
            up2  = self._upconv2d(self.pool5, dim=256, act='linear', name='up2_1') + self._conv2d(self.pool4, dim=256, act='linear', name='pool4_s')
            self.up2_cw = self._conv2d(up2, dim=256, name='up2_3')

            up4  = self._upconv2d(self.up2_cw, dim=128, act='linear', name='up4_1') + self._conv2d(self.pool3, dim=128, act='linear', name='pool3_s')
            self.up4_cw = self._conv2d(up4, dim=128, name='up4_3')

            up8  = self._upconv2d(self.up4_cw, dim=64,  act='linear', name='up8_1') + self._conv2d(self.pool2, dim=64,  act='linear', name='pool2_s')
            self.up8_cw = self._conv2d(up8, dim=64,  name='up8_2')

            up16 = self._upconv2d(self.up8_cw, dim=32,  act='linear', name='up16_1') + self._conv2d(self.pool1, dim=32,  act='linear', name='pool1_s')
            self.up16_cw = self._conv2d(up16, dim=32,  name='up16_2')

            logits_cw = self._up_bilinear(self.up16_cw, dim=3, shape=(h, w), name='logits')

        # 部屋分類（R）ブランチ
        reuse_rnet = len([v for v in tf.compat.v1.global_variables() if v.name.startswith('RNet')]) > 0
        with tf.compat.v1.variable_scope('RNet', reuse=reuse_rnet):
            up2  = self._upconv2d(self.pool5, dim=256, act='linear', name='up2_1') + self._conv2d(self.pool4, dim=256, act='linear', name='pool4_s')
            up2  = self._conv2d(up2, dim=256, name='up2_2')
            up2, _ = self._non_local_context(self.up2_cw, up2, name='context_up2')

            up4  = self._upconv2d(up2, dim=128, act='linear', name='up4_1') + self._conv2d(self.pool3, dim=128, act='linear', name='pool3_s')
            up4  = self._conv2d(up4, dim=128, name='up4_2')
            up4, _ = self._non_local_context(self.up4_cw, up4, name='context_up4')

            up8  = self._upconv2d(up4, dim=64,  act='linear', name='up8_1') + self._conv2d(self.pool2, dim=64,  act='linear', name='pool2_s')
            up8  = self._conv2d(up8, dim=64,  name='up8_2')
            up8, _ = self._non_local_context(self.up8_cw, up8, name='context_up8')

            up16 = self._upconv2d(up8, dim=32,  act='linear', name='up16_1') + self._conv2d(self.pool1, dim=32,  act='linear', name='pool1_s')
            up16 = self._conv2d(up16, dim=32,  name='up16_2')
            self.up16_r, self.a = self._non_local_context(self.up16_cw, up16, name='context_up16')

            logits_r = self._up_bilinear(self.up16_r, dim=9, shape=(h, w), name='logits')

            return logits_r, logits_cw
