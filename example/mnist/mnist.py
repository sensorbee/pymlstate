#!/usr/bin/env python
"""Chainer example: train a multi-layer perceptron on MNIST

This is a minimal example to write a feed-forward net.

"""
import numpy as np
import chainer
from chainer import cuda
from chainer import optimizers
import chainer.links as L
import six

import net  # from chaner MNIST example


class MNIST(object):

    @staticmethod
    def create(n_in=784, n_units=1000, n_out=10, gpu=-1):
        self = MNIST()

        model = L.Classifier(net.MnistMLP(n_in, n_units, n_out))
        if gpu >= 0:
            cuda.get_device(gpu).use()
            model.to_gpu()
        self.model = model

        self.xp = np if gpu < 0 else cuda.cupy

        optimizer = optimizers.Adam()
        optimizer.setup(model)
        self.optimizer = optimizer

        return self

    @staticmethod
    def load(filepath, *args, **kwargs):
        with open(filepath, 'r') as f:
            return six.moves.cPickle.load(f)

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
        x = chainer.Variable(self.xp.asarray(nx, volatile='on'))
        loss = self.model(x)
        y = loss.data.reshape(
            loss.data.shape[0], loss.data.size / loss.data.shape[0])
        pred = y.argmax(axis=1)

        return int(pred[0])

    def save(self, filepath, *args, **kwargs):
        with open(filepath, 'w') as f:
            six.moves.cPickle.dump(self, f)
