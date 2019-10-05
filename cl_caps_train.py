from __future__ import print_function
import tensorflow as tf
import HSI_Data_Preparation
from HSI_Data_Preparation import num_train, Band, All_data, TrainIndex, TestIndex, Height, Width, num_test, Num_Classes
from utils import patch_size, Post_Processing, normalize
import numpy as np
import os
import scipy.io
import time
import capslayer as cl
from caps_model import caps_net, caps_net_mod, caps_net_3
import argparse

parser = argparse.ArgumentParser(description="Capsule Network on MNIST")
parser.add_argument("-e", "--epochs", default=100, type=int,
					help="The number of epochs you want to train.")
parser.add_argument("-g", "--gpu", default="0", type=str,
					help="Which gpu(s) you want to use.")
parser.add_argument("-d", "--directory", default="./saved_model",
					help="The directory you want to save your model.")
parser.add_argument("-b", "--batch", default=100, type=int,
					help="Set batch size.")
parser.add_argument("-m", "--model", default=2, type=int,
					help="Use which model to train and predict.")
parser.add_argument("-r", "--restore", default=False,
					help="Restore the trained model or not. True or False")
parser.add_argument("-c", "--cost", default="margin",
					help="Use margin loss or cross entropy as loss function. 'margin' for margin loss or 'cross' for cross entropy.")
parser.add_argument("-l", "--lr", default=0.001, type=float,
					help="Learning rate")
parser.add_argument("-t", "--predict_times", default=400, type=int,
					help="Times you want to predict. smaller times causes bigger batches.")
parser.add_argument("-o", "--optimizer", default="adam", type=str,
					help="Use adam optimizer or sgd optmizer")
parser.add_argument("-p", "--predict_batch", default=100, type=int,
					help="Predict batch size.")
parser.add_argument("-n", "--normalize", default=1, type=int,
					help="Normalize the result with 1 or not with 0.")
args = parser.parse_args()

print(np.array(All_data["patch"]).shape)

print(args)

os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

start_time = time.time()

training_data, testing_data = HSI_Data_Preparation.Prepare_data()
n_input = Band * patch_size * patch_size

training_data['train_patch'] = np.transpose(training_data['train_patch'], (0, 2, 3, 1))
testing_data['test_patch'] = np.transpose(testing_data['test_patch'], (0, 2, 3, 1))

training_data['train_patch'] = np.reshape(training_data['train_patch'], (-1, n_input))
testing_data['test_patch'] = np.reshape(testing_data['test_patch'], (-1, n_input))

# Parameters
learning_rate = args.lr
batch_size = args.batch
display_step = 500
n_classes = Num_Classes

x = tf.placeholder(shape=[None, n_input], dtype=tf.float32, name="X")
y = tf.placeholder(shape=[None, n_classes], dtype=tf.float32, name="y")

# construct model.
if args.model == 1:
	pred = caps_net(x)
else:
	if args.model == 2:
		pred = caps_net_mod(x)
	else:
		pred = caps_net_3(x)

if args.normalize == 1:
	pred = tf.divide(pred, tf.reduce_sum(pred, 1, keep_dims=True))

margin_loss = cl.losses.margin_loss(y, pred)
cost = tf.reduce_mean(margin_loss)

if args.optimizer == "adam":
	optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)
	print("adam used")
else:
	optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate).minimize(cost)
	print("sgd used")
# Define accuracy
correct_prediction = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))

predict_test_label = tf.argmax(pred, 1)

# Initializing the variables
init = tf.global_variables_initializer()

x_test, y_test = testing_data['test_patch'], testing_data['test_labels']
y_test_scalar = np.argmax(y_test, 1) + 1
x_train, y_train = training_data['train_patch'], training_data['train_labels']

if not os.path.exists(args.directory):
	os.makedirs(args.directory)
save_path = os.path.join(args.directory, "saved_model.ckpt")

# Launch the graph
saver = tf.train.Saver()
with tf.Session() as sess:
	least_loss = 100.0
	
	if args.restore and os.path.exists(args.directory):
		saver.restore(sess, save_path)
		print()
		print("model restored.")
		print()
	else:
		sess.run(init)
	# Training cycle
	
	for epoch in range(args.epochs):
		if epoch % 5 == 0:
			permutation = np.random.permutation(training_data["train_patch"].shape[0])
			training_data["train_patch"] = training_data["train_patch"][permutation, :]
			training_data["train_labels"] = training_data["train_labels"][permutation, :]
			permutation = np.random.permutation(testing_data["test_patch"].shape[0])
			testing_data["test_patch"] = testing_data["test_patch"][permutation, :]
			testing_data["test_labels"] = testing_data["test_labels"][permutation, :]
			print("randomized")
		
		iters = num_train // batch_size
		for iter in range(iters):
			batch_x = training_data['train_patch'][iter * batch_size:(iter + 1) * batch_size, :]
			batch_y = training_data['train_labels'][iter * batch_size:(iter + 1) * batch_size, :]
			_, batch_cost, train_acc = sess.run([optimizer, cost, accuracy], feed_dict={x: batch_x, y: batch_y})
			print("\repochs:{:3d}  batch:{:4d}/{:4d}({:.3f}%)  accuracy:{:.6f}  cost:{:.6f}".format(epoch + 1,
																									iter + 1, iters, (
																												iter + 1) * 100.0 / iters,
																									train_acc,
																									batch_cost), end="")
		print()
		if batch_cost < least_loss:
			least_loss = batch_cost
			saver.save(sess, save_path=save_path)
			print("model saved")
		
		if num_train % batch_size != 0:
			batch_x = training_data['train_patch'][iters * batch_size:, :]
			batch_y = training_data['train_labels'][iters * batch_size:, :]
			_, batch_cost, train_acc = sess.run([optimizer, cost, accuracy], feed_dict={x: batch_x, y: batch_y})
		
		idx = np.random.choice(num_test, size=batch_size, replace=False)
		# Use the random index to select random images and labels.
		test_batch_x = testing_data['test_patch'][idx, :]
		test_batch_y = testing_data['test_labels'][idx, :]
		ac, cs, pr, ory = sess.run([accuracy, cost, pred, y], feed_dict={x: test_batch_x, y: test_batch_y})
		print('Test Data Eval: Test Accuracy = %.4f, Test Cost =%.4f' % (ac, cs))
		
		# pr = normalize(pr)
		#
		for ii in pr[1]:
			print("%.6f" % ii, end=" ")
		print()
		for ii in ory[1]:
			print("%8d" % ii, end=" ")
		print()
	
	print("optimization finished!")
	
	print("==========training data===========")
	arr, rst = sess.run([pred, y],
						feed_dict={x: training_data["train_patch"][0:20, :], y: training_data["train_labels"][0:20, :]})
	
	# arr = normalize(arr)
	
	for i in range(20):
		for item in arr[i]:
			print("%.6f" % item, end=" ")
		print()
		for item in rst[i]:
			print("%8d" % item, end=" ")
		print()
	print()
	
	print("==========test data===========")
	arr, rst = sess.run([pred, y],
						feed_dict={x: testing_data["test_patch"][0:20, :], y: testing_data["test_labels"][0:20, :]})
	for i in range(20):
		for item in arr[i]:
			print("%.6f" % item, end=" ")
		print()
		for item in rst[i]:
			print("%8d" % item, end=" ")
		print()
	print()
	
	# Obtain the probabilistic map
	num_all = len(All_data["patch"])
	pred_times = num_all // args.predict_batch
	prob_map = np.zeros((1, n_classes))
	for i in range(pred_times):
		feedx = np.transpose(np.asarray(All_data["patch"][i * args.predict_batch:(i + 1) * args.predict_batch]),
							 (0, 2, 3, 1))
		feedx = np.reshape(feedx, (-1, n_input))
		temp = sess.run(pred, feed_dict={x: feedx})
		print("\r{:.4f}% data processed.".format((i + 1) * 100.0 / pred_times), end="")
		prob_map = np.concatenate((prob_map, temp), axis=0)
	if num_all % args.predict_batch != 0:
		feedx = np.transpose(np.asarray(All_data["patch"][pred_times * args.predict_batch:]), (0, 2, 3, 1))
		feedx = np.reshape(feedx, (-1, n_input))
		temp = sess.run(pred, feed_dict={x: feedx})
		prob_map = np.concatenate((prob_map, temp), axis=0)
	
	prob_map = np.delete(prob_map, (0), axis=0)
	
	# MRF
	DATA_PATH = os.path.join(os.getcwd(), args.directory)
	
	f = open(os.path.join(DATA_PATH, "101data.txt"), "w+")
	for items in prob_map:
		for i in items:
			print("%.6f" % i, file=f, end=" ")
		print(file=f)
	f.close()
	
	print('The shape of prob_map is (%d,%d)' % (prob_map.shape[0], prob_map.shape[1]))
	arr_test = np.zeros((1, patch_size * patch_size * 220))
	file_name = 'prob_map.mat'
	prob = {}
	prob['prob_map'] = prob_map
	scipy.io.savemat(os.path.join(DATA_PATH, file_name), prob)
	
	train_ind = {}
	train_ind['TrainIndex'] = TrainIndex
	scipy.io.savemat(os.path.join(DATA_PATH, 'TrainIndex.mat'), train_ind)
	
	test_ind = {}
	test_ind['TestIndex'] = TestIndex
	scipy.io.savemat(os.path.join(DATA_PATH, 'TestIndex.mat'), test_ind)
	
	end_time = time.time()
	print('The elapsed time is %.2f' % (end_time - start_time))
	
	f = open(os.path.join(DATA_PATH, "./105label.txt"), "w+")
	for items in All_data["labels"]:
		print(items, file=f)
	f.close()
	
	Seg_Label, seg_Label, seg_accuracy = Post_Processing(prob_map, Height, Width, n_classes, y_test_scalar, TestIndex)
	print("The Final Seg Accuracy is :", seg_accuracy)
