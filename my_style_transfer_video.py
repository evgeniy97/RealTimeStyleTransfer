# import the necessary packages
from imutils.video import VideoStream
from imutils import paths
import itertools
import argparse
import imutils
import time
import cv2
from style_transfer import Transfer
import tensorflow as tf
import utils
import os
import numpy as np
"""
TO DO LIST
	Use Keras/TF to transfer input
	Rewrite code use pip8
"""
checkpoint_dir = r'checkpoints\wave'
checkpoint_dir = r'checkpoints\\the_scream'

# initialize the video stream, then allow the camera sensor to warm up
print("[INFO] starting video stream...")
vs = VideoStream(src=1).start()
time.sleep(2.0)

frame = vs.read()
img_shape = frame.shape
g = tf.Graph()
soft_config = tf.ConfigProto(allow_soft_placement=True)

with g.as_default(), tf.Session(config=soft_config) as sess:
	print("[INFO] starting session...")
	img_placeholder = tf.placeholder(tf.float32, shape=[None, *img_shape], name='img_placeholder')
	model = Transfer()
	pred = model(img_placeholder)
	saver = tf.train.Saver()
	if os.path.isdir(checkpoint_dir):
	    ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
	    if ckpt and ckpt.model_checkpoint_path:
	        saver.restore(sess, ckpt.model_checkpoint_path)
	    else:
	        raise Exception('No checkpoint found...')
	else:
	    saver.restore(sess, checkpoint_dir)
	
	print("[INFO] starting transfer...")
	while True:
		frame = vs.read()
		img = [frame.astype(np.float32)]
		_pred = sess.run(pred, feed_dict={img_placeholder: img})

		cv2.imshow("Input", frame)
		cv2.imshow("Output", np.clip(_pred[0], 0, 255).astype(np.uint8))

		key = cv2.waitKey(1) & 0xFF
		if key == ord("q"):
			break

print("[INFO] ending session...")
cv2.destroyAllWindows()
vs.stop()