import numpy as np
import random


def genData(num, bias, variance):
    x = np.zeros([num, 2])
    y = np.zeros([num])

    for i in range(num):
        x[i][0] = 1
        x[i][1] = i
        y[i] = (i + bias) + random.uniform(0, 1) * variance
    return x, y


def gradientDescent(x, y, theta, alpha, numberIterations):
    xTran = np.transpose(x)
    shape = np.shape(x)
    for i in range(numberIterations):
        hypothesis = np.dot(x, theta)
        loss = hypothesis - y
        cost = np.sum(loss ** 2) / (2 * shape[0])
        gradient = np.dot(xTran, loss) / shape[0]
        theta -= alpha * gradient
        print("Iteration %d | cost :%f" % (i, cost))
    return theta


x, y = genData(100, 25, 10)
numIterations = 100000
alpha = 0.0005
theta = np.ones(2)
theta = gradientDescent(x, y, theta, alpha, numIterations)
print(theta)
