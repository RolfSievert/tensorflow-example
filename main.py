#! /usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2019 hitsnapper <hitsnapper@Znap>
#
# Distributed under terms of the MIT license.

"""
Code example for image classification.
Use pillow for image distortion.
"""

"""
Typical CNN structure:
8x(
    convolutional layer (10 filters, filter size 8, stride 1, padding true)
    relu
    max_pool (pool size 2, stride 2, padding true)
    )

batch norm
relu
dropout
dense (100)

batch norm
relu
dropout
dense (60)

batch norm
relu
dropout
dense (20)

loss function: softmax
learning rate: exponential decay
optimmizer: adam
"""

import tensorflow as tf
import numpy as np
# import argparse

def build_model(X, y):
    """
    Creates a model.
    :X: input layer
    :y: output layer
    """
    return

def run(epochs=1):
    """
    Runs a model.

    :epochs: used as count for number of training steps for model.
    """

    # Create distorted data from data set (create data batch)
    # TODO

    for e in range(epochs):
        sess.run(data);

if __name__ == "__main__":
    # image resolution
    img_res = (500, 500, 3)
    # Input layer
    X = tf.placeholder(tf.float32, [None, *img_res])
    y = tf.placeholder(tf.int64, [None])

    # assign loss function, learning rate and optimizer
    loss_fn = tmp
    learning_rate = tmp
    optimizer = tmp

    # batch norm requires extra dependency
    with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS):
        train_step = optimizer.minimize(mean_loss)

    # Saver saves variables from training
    saver = tf.train.Saver()

    with tf.Session() as sess:
        # Generate training data set from tfrecords-file

        # Generate validation data set from tfrecords-file

        # Run session
        sess.run(tf.global_variables_initializer())

        # Training

        # Validation

        # Save model
        save_path = saver.save(sess, "models/model.ckpt")
