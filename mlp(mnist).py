import tensorflow as tf
import numpy as np
import time
import matplotlib.pyplot as plt
from collections import OrderedDict


def softmax(x):
    # exp_a = np.exp(x)
    # sum_exp_a = np.sum(exp_a)
    # return exp_a / sum_exp_a

    if x.ndim == 2:
        x = x.T
        x = x - np.max(x, axis=0)
        y = np.exp(x) / np.sum(np.exp(x), axis=0)
        return y.T

    x = x - np.max(x)
    return np.exp(x) / np.sum(np.exp(x))

    # c = np.max(x)
    # exp_a = np.exp(x - c)
    # sum_exp_a = np.sum(exp_a)
    # return exp_a / sum_exp_a


def mse(y, t):
    return 0.5 * np.sum((y - t)**2)


def cross_entropy(y, t):
    delta = 1e-7
    return -np.sum(t * np.log(y + delta))


def one_hot_encoding(x):
    arr = np.zeros((x.size, 10))
    for idx, row in enumerate(arr):
        row[x[idx]] = 1

    return arr


class Affine:
    def __init__(self, w, b):
        self.w = w
        self.b = b
        self.x = None
        self.dw = None
        self.db = None

    def forward(self, x):
        self.x = x
        out = np.dot(x, self.w) + self.b
        return out

    def backward(self, dout):
        dx = np.dot(dout, self.w.T)
        self.dw = np.dot(self.x.T, dout)
        self.db = np.sum(dout, axis=0)
        return dx


class Sigmoid:
    def __init__(self):
        self.out = None

    def forward(self, x):
        out = 1 / (1 + np.exp(-x))
        self.out = out
        return out

    def backward(self, dout):
        dx = dout * (1.0 - self.out) * self.out
        return dx


class SoftmaxWithLoss:
    def __init__(self):
        self.loss = None
        self.y = None
        self.t = None

    def forward(self, x, t):
        self.t = t
        self.y = softmax(x)
        self.loss = cross_entropy(self.y, self.t)
        # self.loss = mse(self.y, self.t)
        return self.loss

    def backward(self, dout=1):
        batch_size = self.t.shape[0]
        dx = (self.y - self.t) / batch_size
        return dx

class Network:
    def __init__(self, input_size, hidden_size, output_size, weight_init_std=0.01):
        self.params = {}
        self.params["w1"] = weight_init_std * np.random.randn(input_size, hidden_size)
        self.params["b1"] = np.zeros(hidden_size)
        self.params["w2"] = weight_init_std * np.random.randn(hidden_size, output_size)
        self.params["b2"] = np.zeros(output_size)

        self.layers = OrderedDict()
        self.layers["Affine1"] = Affine(self.params["w1"], self.params["b1"])
        self.layers["Sigmoid"] = Sigmoid()
        self.layers["Affine2"] = Affine(self.params["w2"], self.params["b2"])
        self.lastLayer = SoftmaxWithLoss()

    def predict(self, x):
        for layer in self.layers.values():
            x = layer.forward(x)
        return x

    def loss(self, x, t):
        y = self.predict(x)
        return self.lastLayer.forward(y, t)

    def accuracy(self, x, t):
        y = self.predict(x)
        y = np.argmax(y, axis=1)
        t = np.argmax(t, axis=1)
        acc = np.sum(y == t) / float(x.shape[0])
        return acc

    def gradient(self, x, t):
        self.loss(x, t)

        dout = 1
        dout = self.lastLayer.backward(dout)
        layers = list(self.layers.values())
        layers.reverse()

        for layer in layers:
            dout = layer.backward(dout)

        grads = {}
        grads["w1"] = self.layers["Affine1"].dw
        grads["b1"] = self.layers["Affine1"].db
        grads["w2"] = self.layers["Affine2"].dw
        grads["b2"] = self.layers["Affine2"].db
        return grads


mnist = tf.keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()


network = Network(input_size=784, hidden_size=50, output_size=10)
epoch = 20
train_size = x_train.shape[0]
batch_size = 60
train_loss = []
train_accuracy = []
test_accuracy = []
iteration = int(train_size / batch_size)   # 1000
lr = 0.01

print(x_train.shape)   # (60000, 28, 28)
print(y_train.shape)   # (60000, )
print(x_test.shape)   # (10000, 28, 28)
print(y_test.shape)   # (10000, )
print(x_train[0].shape)   # (28, 28)
print(y_train[0])   # 5

x_train = x_train.reshape(60000, 784)
print(x_train.shape)   # (60000, 784 )
x_test = x_test.reshape(10000, 784)
print(x_test.shape)   # (10000, 784 )

y_train = one_hot_encoding(y_train)
y_test = one_hot_encoding(y_test)

start = time.time()

for i in range(epoch):
    x_batch = None
    y_batch = None
    for j in range(iteration):
        batch_mask = np.random.choice(train_size, batch_size, replace=False)  # 비복원 추출
        x_batch = x_train[batch_mask]
        y_batch = y_train[batch_mask]

        grad = network.gradient(x_batch, y_batch)

        for key in ("w1", "b1", "w2", "b2"):
            network.params[key] -= lr * grad[key]   # SGD

    loss = network.loss(x_batch, y_batch)
    train_loss.append(loss)

    train_acc = network.accuracy(x_batch, y_batch)
    train_accuracy.append(train_acc)

    test_acc = network.accuracy(x_test, y_test)
    test_accuracy.append(test_acc)

    print("Epoch {} : train_accuracy : {:5.5f}, train_loss : {:5.5f}, test_accuracy : {:5.5f}".format(i + 1, train_acc, loss, test_acc))

end = time.time()
print("\n실행 시간(mse) : {:.3f}".format(end - start))


ax1 = plt.subplot(2, 1, 1)
plt.plot(np.arange(epoch), train_loss, "o-", c="red", label="train_loss")
plt.ylabel("loss")
plt.xticks(visible=False)
plt.legend(loc="upper right")

ax2 = plt.subplot(2, 1, 2, sharex=ax1)
plt.plot(np.arange(epoch), train_accuracy, "o-", c="blue", label="train_acc")
plt.plot(np.arange(epoch), test_accuracy, "o-", c="black", label="test_accuracy")
plt.xticks(np.arange(epoch))
plt.xlabel("epoch")
plt.ylabel("accuracy")

plt.legend(loc="upper right")
plt.show()
