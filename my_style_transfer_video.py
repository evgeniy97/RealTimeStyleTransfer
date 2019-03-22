# import the necessary packages
from imutils.video import VideoStream
import glob 
import itertools
import time
import cv2
from style_transfer import Transfer
import tensorflow as tf
import os
import numpy as np
"""
TO DO LIST
	Rewrite code use pip8
	Learn which styles we have
	Take photo
"""

def load(saver, checkpoint_dir):
	if os.path.isdir(checkpoint_dir):
		ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
		if ckpt and ckpt.model_checkpoint_path:
			saver.restore(sess, ckpt.model_checkpoint_path)
		else:
			raise Exception('No checkpoint found...')
	else:
	    saver.restore(sess, checkpoint_dir)


def saverPicture():
	pass


# initialize the video stream, then allow the camera sensor to warm up

style = glob.glob('checkpoints/*')
styleIter = itertools.cycle(style) 
checkpoint_dir = next(styleIter)


print("[INFO] starting video stream...")
vs = VideoStream(src=1).start()
time.sleep(2.0)

frame = vs.read()
img_shape = frame.shape
g = tf.Graph()
soft_config = tf.ConfigProto(allow_soft_placement=True)
soft_config.gpu_options.allow_growth = True

with g.as_default(), tf.Session(config=soft_config) as sess:
	print("[INFO] starting session...")
	img_placeholder = tf.placeholder(tf.float32, shape=[None, *img_shape], name='img_placeholder')
	model = Transfer()
	pred = model(img_placeholder)
	saver = tf.train.Saver()
	
	load(saver, checkpoint_dir)

	print("[INFO] starting transfer...")
	while True:
		frame = vs.read()
		img = [frame.astype(np.float32)]
		_pred = sess.run(pred, feed_dict={img_placeholder: img})

		_pred = np.clip(_pred[0], 0, 255).astype(np.uint8)

		cv2.imshow("Input", frame)
		cv2.imshow("Output", _pred)

		key = cv2.waitKey(1) & 0xFF
		
		if key == ord("n"):
			checkpoint_dir = next(styleIter)
			load(saver, checkpoint_dir)
		elif key == ord("s"):
			cv2.imwrite('photo.jpg',_pred)
		elif key == ord("q"):
			break

print("[INFO] ending session...")
cv2.destroyAllWindows()
vs.stop()