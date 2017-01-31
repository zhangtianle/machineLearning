import numpy as np
import random

def kmeans(x, k, maxIt):
    num, dim = np.shape(x)
    iterations = 0
    dataSet = np.zeros((num, dim + 1))
    dataSet[:, :-1] = x

    centroids = dataSet[random.sample(range(num), k), :]
    centroids[:, -1] = range(k)
    oldCentroids = None

    while not stop(oldCentroids, centroids, iterations, maxIt):
        iterations += 1
        oldCentroids = np.copy(centroids)

        updateLables(dataSet, centroids)
        centroids = getCentroids(dataSet, k)
        # print(dataSet)
    return dataSet

def stop(oldCentroids, centroids, iterations, maxIt):
    if iterations > maxIt:
        return True
    return np.array_equal(oldCentroids, centroids)

def updateLables(dataSet, centroids):
    num, dim = np.shape(dataSet)
    for i in range(num):
        dataSet[i, -1] = getLableFromClosestCentroid(dataSet[i, :-1], centroids)

def getLableFromClosestCentroid(param, centroids):
    num, dim = np.shape(centroids)
    lable = centroids[0, -1]
    minDis = np.linalg.norm(param - centroids[0, :-1])
    for i in range(num):
        dis = np.linalg.norm(param - centroids[i, :-1])
        if dis < minDis:
            lable = centroids[i, -1]
            minDis = dis
    return lable

def getCentroids(dataSet, k):
    result = np.zeros((k, dataSet.shape[1]))
    for i in range(k):
        temp = dataSet[dataSet[:, -1] == i, : -1]
        result[i, :-1] = np.mean(temp, axis=0)
        result[i, -1] = i
    return result

x = np.array([[1, 1], [2, 1], [4, 3], [5, 4]])
result = kmeans(x, 2, 100)
print(result)