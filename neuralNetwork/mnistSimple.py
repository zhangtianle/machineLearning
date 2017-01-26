from sklearn.datasets import load_digits
from sklearn.cross_validation import train_test_split
from sklearn.preprocessing import LabelBinarizer
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
import numpy as np
from NeuralNetwork import NeuralNetwork

digits = load_digits()
x = digits.data
y = digits.target
x -= x.min()
x /= x.max()

# plt.gray()
# plt.matshow(digits.images[0])
# plt.show()

nn = NeuralNetwork([64, 100, 10], "sigmoid")
x_train, x_test, y_train, y_test = train_test_split(x, y)
label_train = LabelBinarizer().fit_transform(y_train)
label_test = LabelBinarizer().fit_transform(y_test)
predictions = []
nn.fit(x_train, label_train, epochs=10000)
for i in range(x_test.shape[0]):
    o = nn.predict(x_test[i])
    predictions.append(np.argmax(o))
print(confusion_matrix(y_test, predictions))
print(classification_report(y_test, predictions))