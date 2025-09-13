import os
import argparse
import numpy as np
import tensorflow as tf

from matplotlib import pyplot as plt

from imageio.v2 import imread, imwrite as imsave  # imread/imsave はこれでOK
from PIL import Image

def imresize(img, size, interp='bilinear', mode=None):
    """
    A lightweight imresize compatible with the old API:
      - size: float(倍率) or (H, W)
      - interp: 'nearest' | 'bilinear' | 'bicubic' | 'lanczos'
    dtype/範囲は元配列に合わせて返します（float→[0,1]、整数→整数）。
    """
    # --- to PIL image (uint8, proper mode) ---
    orig_dtype = img.dtype
    if img.ndim == 2:
        pil_mode = 'L'
    elif img.ndim == 3 and img.shape[2] == 3:
        pil_mode = 'RGB'
    elif img.ndim == 3 and img.shape[2] == 4:
        pil_mode = 'RGBA'
    else:
        # それ以外の形状は未対応（必要なら拡張してください）
        raise ValueError(f"Unsupported image shape for imresize: {img.shape}")

    if np.issubdtype(orig_dtype, np.floating):
        src = np.clip(img, 0.0, 1.0) * 255.0
        src = src.astype(np.uint8)
    else:
        src = np.clip(img, 0, 255).astype(np.uint8)

    pil_img = Image.fromarray(src, mode=pil_mode)

    # --- resolve target size ---
    if isinstance(size, (int, float)):
        new_w = int(round(pil_img.width * float(size)))
        new_h = int(round(pil_img.height * float(size)))
        target = (new_w, new_h)
    elif isinstance(size, (tuple, list)) and len(size) == 2:
        # old imresize は (H, W)。Pillow は (W, H) なので入れ替え
        target = (int(size[1]), int(size[0]))
    else:
        raise ValueError(f"Invalid size for imresize: {size}")

    # --- interpolation map ---
    _imap = {
        'nearest': Image.NEAREST,
        'bilinear': Image.BILINEAR,
        'bicubic': Image.BICUBIC,
        'lanczos': Image.LANCZOS,
    }
    resample = _imap.get(interp, Image.BILINEAR)

    # --- resize & back to numpy with original dtype/range ---
    out = pil_img.resize(target, resample=resample)
    arr = np.array(out)

    if np.issubdtype(orig_dtype, np.floating):
        arr = (arr.astype(np.float32) / 255.0).astype(orig_dtype)
    else:
        arr = arr.astype(orig_dtype)
    return arr
	
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# input image path
parser = argparse.ArgumentParser()

parser.add_argument('--im_path', type=str, default='./demo/45765448.jpg',
                    help='input image paths.')

# color map
floorplan_map = {
	0: [255,255,255], # background
	1: [192,192,224], # closet
	2: [192,255,255], # batchroom/washroom
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
	rgb_im = np.zeros((ind_im.shape[0], ind_im.shape[1], 3))

	for i, rgb in color_map.iteritems():
		rgb_im[(ind_im==i)] = rgb

	return rgb_im

def main(args):
	# load input
	im = imread(args.im_path, mode='RGB')
	im = im.astype(np.float32)
	im = imresize(im, (512,512,3)) / 255.

	# create tensorflow session
	with tf.Session() as sess:
		
		# initialize
		sess.run(tf.group(tf.global_variables_initializer(),
					tf.local_variables_initializer()))

		# restore pretrained model
		saver = tf.train.import_meta_graph('./pretrained/pretrained_r3d.meta')
		saver.restore(sess, './pretrained/pretrained_r3d')

		# get default graph
		graph = tf.get_default_graph()

		# restore inputs & outpus tensor
		x = graph.get_tensor_by_name('inputs:0')
		room_type_logit = graph.get_tensor_by_name('Cast:0')
		room_boundary_logit = graph.get_tensor_by_name('Cast_1:0')

		# infer results
		[room_type, room_boundary] = sess.run([room_type_logit, room_boundary_logit],\
										feed_dict={x:im.reshape(1,512,512,3)})
		room_type, room_boundary = np.squeeze(room_type), np.squeeze(room_boundary)

		# merge results
		floorplan = room_type.copy()
		floorplan[room_boundary==1] = 9
		floorplan[room_boundary==2] = 10
		floorplan_rgb = ind2rgb(floorplan)

		# plot results
		plt.subplot(121)
		plt.imshow(im)
		plt.subplot(122)
		plt.imshow(floorplan_rgb/255.)
		plt.show()

if __name__ == '__main__':
	FLAGS, unparsed = parser.parse_known_args()
	main(FLAGS)
