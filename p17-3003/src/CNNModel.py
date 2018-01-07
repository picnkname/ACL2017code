import tensorflow as tf
import tflearn
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.estimator import regression

import os
os.environ['CUDA_VISIBLE_DEVICES'] = '1'


def build_model(input_size, input_channels, output_size=2, keep_prob=0.8):
    network = input_data([None, input_size, input_size, input_channels], name='input')
    network = conv_2d(network, 12, 3, activation='relu', name='conv1')
    network = max_pool_2d(network, 2, 2, name='max_pool_1')
    network = conv_2d(network, 16, 3, activation='relu', name='conv2')
    network = max_pool_2d(network, 2, 2, name='max_pool_2')
    network = fully_connected(network, 200, activation='relu')
    network = dropout(network, keep_prob, name='dropout1')
    network = fully_connected(network, output_size, activation='softmax', name='output')
    network = regression(network, optimizer='adam', learning_rate=0.0005, batch_size=5,
                         loss='categorical_crossentropy', name='target')
    model = tflearn.DNN(network, tensorboard_verbose=0)
    return model


if __name__ == '__main__':
    pass


