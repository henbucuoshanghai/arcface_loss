import tensorflow as tf
import math
import numpy as np
INPUT_NODE = 784
OUTPUT_NODE = 10

IMAGE_SIZE = 28
NUM_CHANNELS = 1
NUM_LABELS = 10

CONV1_DEEP = 32
CONV1_SIZE = 5

CONV2_DEEP = 64
CONV2_SIZE = 5

FC_SIZE = 512

def inference(input_tensor, train, regularizer,train_label):
	with tf.variable_scope('layer1-conv1'):
		conv1_weights = tf.get_variable(
			"weight", [CONV1_SIZE, CONV1_SIZE, NUM_CHANNELS, CONV1_DEEP],
			initializer=tf.truncated_normal_initializer(stddev=0.1))
		conv1_biases = tf.get_variable("bias", [CONV1_DEEP], initializer=tf.constant_initializer(0.0))
		conv1 = tf.nn.conv2d(input_tensor, conv1_weights, strides=[1, 1, 1, 1], padding='SAME')
             #   conv1=tf.layers.batch_normalization(conv1, is_training)
		relu1 = tf.nn.relu(tf.nn.bias_add(conv1, conv1_biases))

	with tf.name_scope("layer2-pool1"):
		pool1 = tf.nn.max_pool(relu1, ksize = [1,2,2,1],strides=[1,2,2,1],padding="SAME")

	with tf.variable_scope("layer3-conv2"):
		conv2_weights = tf.get_variable(
			"weight", [CONV2_SIZE, CONV2_SIZE, CONV1_DEEP, CONV2_DEEP],
			initializer=tf.truncated_normal_initializer(stddev=0.1))
		conv2_biases = tf.get_variable("bias", [CONV2_DEEP], initializer=tf.constant_initializer(0.0))
		conv2 = tf.nn.conv2d(pool1, conv2_weights, strides=[1, 1, 1, 1], padding='SAME')
              #  conv2=tf.layers.batch_normalization(conv2, is_training)
		relu2 = tf.nn.relu(tf.nn.bias_add(conv2, conv2_biases))

	with tf.name_scope("layer4-pool2"):
		pool2 = tf.nn.max_pool(relu2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
		pool_shape = pool2.get_shape()
		nodes = pool_shape[1] * pool_shape[2] * pool_shape[3]
		#reshaped = tf.reshape(pool2, [pool_shape[0], nodes])
		reshaped = tf.reshape(pool2, [-1, nodes])

	with tf.variable_scope('layer5-fc1'):
		fc1_weights = tf.get_variable("weight", [nodes, FC_SIZE],
									  initializer=tf.truncated_normal_initializer(stddev=0.1))
		if regularizer != None: tf.add_to_collection('losses', regularizer(fc1_weights))
		fc1_biases = tf.get_variable("bias", [FC_SIZE], initializer=tf.constant_initializer(0.1))

		fc1 = tf.nn.relu(tf.matmul(reshaped, fc1_weights) + fc1_biases)
		if train: fc1 = tf.nn.dropout(fc1, 0.7)

	with tf.variable_scope('layer6-fc2'):
		fc2_weights = tf.get_variable("weight", [FC_SIZE, NUM_LABELS],
									  initializer=tf.truncated_normal_initializer(stddev=0.1))
		if regularizer != None: tf.add_to_collection('losses', regularizer(fc2_weights))
		
		x=fc1
		w=fc2_weights
		margin=0.5
		scale=64
	        cos_m = math.cos(margin)
        	sin_m = math.sin(margin)
       	 	normed_weights = tf.nn.l2_normalize(w, 1, 1e-10, name='weights_norm')
        	normed_features = tf.nn.l2_normalize(x, 1, 1e-10, name='features_norm')

        	cosine = tf.matmul(normed_features, normed_weights)
        	one_hot_mask = tf.one_hot(train_label,10, on_value=1., off_value=0., axis=-1, dtype=tf.float32)

        	cosine_theta_2 = tf.pow(cosine, 2., name='cosine_theta_2')
        	sine_theta = tf.pow(1. - cosine_theta_2, .5, name='sine_theta')
        	cosine_theta_m = scale * (cos_m * cosine - sin_m * sine_theta) * one_hot_mask
		
		clip_mask = tf.to_float(cosine >= 0.) * scale * cosine * one_hot_mask
		cosine = scale * cosine * (1. - one_hot_mask)+tf.where(clip_mask > 0., cosine_theta_m, clip_mask)
        	return cosine,tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=train_label,
                                                                            logits=cosine), name='arc_loss')
