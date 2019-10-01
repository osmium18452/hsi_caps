import tensorflow as tf
import numpy as np
from utils import squash, patch_size
# from tensorflow.contrib import slim
import capslayer as cl

num_band = 220  # paviaU 103
num_classes = 16


def caps_net(x):
	net = tf.reshape(x, [-1, patch_size, patch_size, num_band])
	conv1 = tf.layers.conv2d(
		net,
		filters=32,
		kernel_size=3,
		strides=1,
		padding="VALID",
		activation=tf.nn.relu,
		name="convLayer"
	)
	
	convCaps, activation = cl.layers.primaryCaps(
		conv1,
		filters=32,
		kernel_size=3,
		strides=1,
		out_caps_dims=[8, 1],
		method="logistic"
	)
	
	n_input = np.prod(cl.shape(convCaps)[1:4])
	convCaps = tf.reshape(convCaps, shape=[-1, n_input, 8, 1])
	activation = tf.reshape(activation, shape=[-1, n_input])
	
	rt_poses, rt_probs = cl.layers.dense(
		convCaps,
		activation,
		num_outputs=num_classes,
		out_caps_dims=[16, 1],
		routing_method="DynamicRouting"
	)
	return rt_probs


def caps_net_mod(x):
	net = tf.reshape(x, [-1, patch_size, patch_size, num_band])
	conv1 = tf.layers.conv2d(
		net,
		filters=100,
		kernel_size=3,
		strides=1,
		padding="same",
		activation=tf.nn.relu,
		name="convLayer"
	)
	conv1 = tf.layers.max_pooling2d(conv1, 2, strides=2, padding="same")
	
	conv2 = tf.layers.conv2d(
		conv1,
		filters=300,
		kernel_size=3,
		padding="same",
		activation=tf.nn.relu
	)
	conv2 = tf.layers.max_pooling2d(conv2, 2, strides=2, padding="same")
	
	convCaps, activation = cl.layers.primaryCaps(
		conv2,
		filters=64,
		kernel_size=3,
		strides=1,
		out_caps_dims=[8, 1],
		method="logistic"
	)
	
	n_input = np.prod(cl.shape(convCaps)[1:4])
	convCaps = tf.reshape(convCaps, shape=[-1, n_input, 8, 1])
	activation = tf.reshape(activation, shape=[-1, n_input])
	
	rt_poses, rt_probs = cl.layers.dense(
		convCaps,
		activation,
		num_outputs=num_classes,
		out_caps_dims=[16, 1],
		routing_method="DynamicRouting"
	)
	return rt_probs
