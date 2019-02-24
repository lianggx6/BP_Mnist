# coding:utf-8
import numpy as np


class NueraLNet(object):
    def __init__(self, sizes):
        self.num_layers = len(sizes)
        self.sizes = sizes
        self.bias = [np.random.randn(1, y) for y in sizes[1:]]
        self.weights = [np.random.randn(x, y) for x, y in zip(sizes[:-1], sizes[1:])]

    def get_result(self, images):
        result = images
        for b, w in zip(self.bias, self.weights):
            result = sigmoid(np.dot(result, w) + b)
        return result

    def train_net(self, trainimage, trainresult, traintime, rate=1, minibatch=10, test_image=None, test_result=None):
        for i in range(traintime):
            minibatchimage = [trainimage[k:k+minibatch] for k in range(0, len(trainimage), minibatch)]
            minibatchresult = [trainresult[k:k+minibatch] for k in range(0, len(trainimage), minibatch)]
            for image, result in zip(minibatchimage, minibatchresult):
                self.update_net(image, result, rate)
            print("第{0}次学习结束".format(i+1))
            if test_image and test_result:
                self.test_net(test_image, test_result)

    def update_net(self, training_image, training_result, rate):
        batch_b_error = [np.zeros(b.shape) for b in self.bias]
        batch_w_error = [np.zeros(w.shape) for w in self.weights]
        for image, result in zip(training_image, training_result):
            b_error, w_error = self.get_error(image, result)
            batch_b_error = [bbe + be for bbe, be in zip(batch_b_error, b_error)]
            batch_w_error = [bwe + we for bwe, we in zip(batch_w_error, w_error)]
        self.bias = [b - (rate/len(training_image))*bbe for b, bbe in zip(self.bias, batch_b_error)]
        self.weights = [w - (rate/len(training_image))*bwe for w, bwe in zip(self.weights, batch_w_error)]

    def get_error(self, image, result):
        b_error = [np.zeros(b.shape) for b in self.bias]
        w_error = [np.zeros(w.shape) for w in self.weights]
        out_data = [image]
        in_data = []
        for b, w in zip(self.bias, self.weights):
            in_data.append(np.dot(out_data[-1], w) + b)
            out_data.append(sigmoid(in_data[-1]))
        b_error[-1] = sigmoid_prime(in_data[-1]) * (out_data[-1] - result)
        w_error[-1] = np.dot(out_data[-2].transpose(), b_error[-1])
        for l in range(2, self.num_layers):
            b_error[-l] = sigmoid_prime(in_data[-l]) * \
                          np.dot(b_error[-l+1], self.weights[-l+1].transpose())
            w_error[-l] = np.dot(out_data[-l-1].transpose(), b_error[-l])
        return b_error, w_error

    def test_net(self, test_image, test_result):
        results = [(np.argmax(self.get_result(image)), result)
                   for image, result in zip(test_image, test_result)]
        right = sum(int(x == y) for (x, y) in results)
        print("正确率：{0}/{1}".format(right, len(test_result)))
        return results

    def save_training(self):
        np.savez('./datafile/weights.npz', *self.weights)
        np.savez('./datafile/bias.npz', *self.bias)

    def read_training(self):
        length = len(self.sizes) - 1
        file_weights = np.load('./datafile/weights.npz')
        file_bias = np.load('./datafile/bias.npz')
        self.weights = []
        self.bias = []
        for i in range(length):
            index = "arr_" + str(i)
            self.weights.append(file_weights[index])
            self.bias.append(file_bias[index])


def sigmoid(x):
    return np.longfloat(1.0 / (1.0 + np.exp(-x)))


def sigmoid_prime(x):
    return sigmoid(x) * (1 - sigmoid(x))
