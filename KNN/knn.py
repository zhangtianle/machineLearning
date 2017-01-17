from sklearn import datasets
from sklearn import neighbors

knn = neighbors.KNeighborsClassifier()

iris = datasets.load_iris()
# print(iris)

knn.fit(iris.data, iris.target)

predictedLable = knn.predict([[8, 2, 3, 2]])

print(predictedLable)