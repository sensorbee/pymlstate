#!/usr/bin/env python
"""Chainer example: train a multi-layer perceptron on MNIST

This is a minimal example to write a feed-forward net.

"""
import numpy as np
import chainer
from chainer import cuda
from chainer import optimizers
from chainer import serializers
import chainer.functions as F
import chainer.links as L


class MNIST(object):

    @staticmethod
    def create(n_in=784, n_units=1000, n_out=10, gpu=-1):
        self = MNIST()

        model = L.Classifier(MnistMLP(n_in, n_units, n_out))
        if gpu >= 0:
            cuda.get_device(gpu).use()
            model.to_gpu()
        self.model = model

        self.gpu = gpu
        self.xp = np if gpu < 0 else cuda.cupy

        optimizer = optimizers.Adam()
        optimizer.setup(model)
        self.optimizer = optimizer

        return self

    @staticmethod
    def load(filepath, n_in=784, n_units=1000, n_out=10, gpu=-1):
        self = MNIST()
        model = L.Classifier(MnistMLP(n_in, n_units, n_out))
        serializers.load_npz(filepath, model)
        if gpu >= 0:
            cuda.get_device(gpu).use()
            model.to_gpu()
        self.model = model

        # To resume learning models, need to save/load optimizer.

        self.gpu = gpu
        self.xp = np if self.gpu < 0 else cuda.cupy

        return self

    def fit(self, xys):
        x_train = []
        y_train = []
        for d in xys:
            x_train.append(d['data'])
            y_train.append(d['label'])

        nx = np.array(x_train, dtype=np.float32)
        ny = np.array(y_train, dtype=np.int32)
        x = chainer.Variable(self.xp.asarray(nx))
        t = chainer.Variable(self.xp.asarray(ny))

        # Pass the loss function (Classifier defines it) and its arguments
        self.optimizer.update(self.model, x, t)

        sum_loss = float(self.model.loss.data) * len(t.data)
        sum_accuracy = float(self.model.accuracy.data) * len(t.data)

        return sum_loss, sum_accuracy

    def predict(self, x):
        x_test = []
        x_test.append(x)
        nx = np.array(x_test, dtype=np.float32)
        xx = chainer.Variable(self.xp.asarray(nx))
        y = self.model.predictor(xx).data
        y = y.reshape(len(y), -1)  # flatten
        pred = y.argmax(axis=1)

        return int(pred[0])

    def save(self, filepath, *args, **kwargs):
        serializers.save_npz(filepath, self.model)


# from chainer/example/mnist/net.py
class MnistMLP(chainer.Chain):

    """An example of multi-layer perceptron for MNIST dataset.

    This is a very simple implementation of an MLP. You can modify this code to
    build your own neural net.

    """
    def __init__(self, n_in, n_units, n_out):
        super(MnistMLP, self).__init__(
            l1=L.Linear(n_in, n_units),
            l2=L.Linear(n_units, n_units),
            l3=L.Linear(n_units, n_out),
        )

    def __call__(self, x):
        h1 = F.relu(self.l1(x))
        h2 = F.relu(self.l2(h1))
        return self.l3(h2)
