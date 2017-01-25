import numpy as np

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoidDeriv(x):
    return sigmoid(x) * (1 - sigmoid(x))

def tanh(x):
    return np.tanh(x)

def tanhDeriv(x):
    return 1.0 - np.tanh(x) * np.tanh(x)

class NeuralNetwork:

    _activation = tanh
    _activationDeriv = tanhDeriv
    _weights = []

    def __init__(self, layers, activation='tanh'):
        if activation == 'tanh':
            self._activation = tanh
            self._activationDeriv = tanhDeriv
        elif activation == 'sigmoid':
            self._activation = sigmoid
            self._activationDeriv = sigmoidDeriv

        # for i in range(1, len(layers)):
        #     self._weights.append(np.random.randn(layers[i]))
        for i in range(1, len(layers) - 1):
            print(layers[i - 1] + 1)
            self._weights.append((2*np.random.random((layers[i - 1] + 1, layers[i] + 1))-1)*0.25)
            self._weights.append((2*np.random.random((layers[i] + 1, layers[i + 1]))-1)*0.25)

    def fit(self, x, y, learningRate=0.2, epochs=10000):
        x = np.atleast_2d(x)
        temp = np.ones([x.shape[0], x.shape[1]+1])
        temp[:, 0:-1] = x
        x = temp

        for k in range(epochs):
            i = np.random.randint(x.shape[0])
            result = [x[i]]
            for l in range(len(self._weights)):
                result.append(self._activation(np.dot(result[l], self._weights[l])))
            error = y[i] - result[-1]
            deltas = [error * self._activationDeriv(result[-1])]

            for l in range(len(self._weights)-1, 0, -1):
                deltas.append(np.dot(self._weights[l], deltas[-1]) * self._activationDeriv(result[l]))
                # deltas.append(deltas[-1].dot(self._weights[l].T) * self._activationDeriv(result[l]))
            deltas.reverse()

            for i in range(len(self._weights)):
                layer = np.atleast_2d(result[i])
                delta = np.atleast_2d(deltas[i])
                self._weights[i] += learningRate * layer.T.dot(delta)

    def predict(self, x):
        x = np.array(x)
        temp = np.ones(x.shape[0] + 1)
        temp[0:-1] = x
        a = temp
        for l in range(0, len(self._weights)):
            a = self._activation(np.dot(a, self._weights[l]))
        return a



