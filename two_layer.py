# coding=utf-8
import numpy as np
import random
from PIL import Image, ImageFilter
import os
import scipy.io
from tensorflow.examples.tutorials.mnist import input_data

def shuffle_data(X, Y):
    index = [i for i in range(len(X))]
    random.shuffle(index)
    return X[index, :], Y[index]

def binary_images(X):
    return (X > 0.5) + 0.0


def relu(x):
    if x > 0.0:
        return x
    else:
        return 0.0


def relu_prime(x):
    if x > 0.0:
        return 1.0
    else:
        return 0.0

relu_vector = np.vectorize(pyfunc=relu)
relu_prime_vector = np.vectorize(pyfunc=relu_prime)


def softmax(Z):
    for i in range(len(Z)):
        m = np.max(Z[i])
        a = np.exp(Z[i] - m)
        Z[i] = a / np.sum(a)
    return Z




# mnist
mnist = input_data.read_data_sets(r'./data/mnist', one_hot=True)

Xtrain = np.concatenate((mnist.train.images, mnist.validation.images), axis=0)
Ytrain = np.concatenate((mnist.train.labels, mnist.validation.labels), axis=0)
Xtrain = binary_images(Xtrain)

Xtest = mnist.test.images
Ytest = mnist.test.labels
Xtest = binary_images(Xtest)





class Network(object):
    def __init__(self, sizes):
        if len(sizes) != 3:
            raise ValueError("网络只能为3层")
        self.W1 = np.random.normal(loc=0.0, scale=0.1, size=sizes[:2])
        self.W2 = np.random.normal(loc=0.0, scale=0.1, size=sizes[1:])

    def backprop(self, batch_X, batch_Y):
        m = len(batch_X)
        # forward
        Z1 = np.dot(batch_X, self.W1)
        A1 = relu_vector(Z1)
        Z2 = np.dot(A1, self.W2)
        A2 = softmax(Z2)

        # backward
        Delta2 = A2 - batch_Y
        dW2 = np.dot(A1.transpose(), Delta2) / m
        Delta1 = np.dot(Delta2, self.W2.transpose()) * relu_prime_vector(Z1)
        dW1 = np.dot(batch_X.transpose(), Delta1) / m

        return dW1, dW2

    def SGD(self, batch_X, batch_Y, lr, lamda=0.1):
        m = len(batch_X)
        dW1, dW2 = self.backprop(batch_X, batch_Y)
        self.W1 -= (lr * dW1 + lr * lamda * self.W1 / m)
        self.W2 -= (lr * dW2 + lr * lamda * self.W2 / m)

    def evaluate(self, test_X, test_Y):
        Z1 = np.dot(test_X, self.W1)
        A1 = relu_vector(Z1)
        Z2 = np.dot(A1, self.W2)
        A2 = softmax(Z2)
        return np.sum(np.argmax(A2, axis=1) == np.argmax(test_Y, axis=1)) / len(test_X)


net = Network([784, 100, 10])

Xs = Xtrain.copy()
Ys = Ytrain.copy()


batch_size = 64
for epoch in range(150):
    lr = 0.1
    if epoch > 50:
        lr = 0.01
    if epoch > 100:
        lr = 0.001


    Xs, Ys = shuffle_data(Xs, Ys)
    for j in range(int(len(Xs)/batch_size)):
        batch_xs = Xs[j * batch_size:(j + 1) * batch_size]
        batch_ys = Ys[j * batch_size:(j + 1) * batch_size]
        net.SGD(batch_xs, batch_ys, lr=lr)
    print("Epoch:", epoch,
          ", Acc_train:", net.evaluate(Xtrain, Ytrain),
          ", Acc_test:", net.evaluate(Xtest, Ytest))

