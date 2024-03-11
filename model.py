import torch as torch
import torch.nn as nn
import torch.nn.functional as F
import tensorflow as tf
from tensorflow.keras import layers

class DQN(tf.keras.Model):
    def __init__(self):
        #worry about which layers to use here
        self.flatten = layers.Flatten()

        self.dense1 = layers.Dense(256, activation='relu')
        self.dense2 = layers.Dense(128, activation='relu')
        self.dense3 = layers.Dense(64, activation='relu')
        self.output = layers.Dense(1, activation=None)

    def forward(self, state):
        x = self.flatten(state)
        x = self.dense1(x)
        x = self.dense2(x)
        x = self.dense3(x)
        value = self.output(x)
        return value