import math

class clusterNode:
    def __init__(self, vec, left=None, right=None, id=None, distance=0.0):
        self.vec = vec
        self.left = left
        self.right = right
        self.id = id
        self.distance = distance


def euclideanDistance(vec1, vec2):
    length = len(vec1)
    TSum = sum([pow((vec1[i] - vec2[i]), 2) for i in range(length)])
    SSum = math.sqrt(TSum)
    return SSum


def yezi(clust):
    if clust.left == None and clust.right == None:
        return [clust.id]
    return yezi(clust.left) + yezi(clust.right)


def hcluster(vec, n):
    clusterNodes = [clusterNode(vec[i], id=i) for i in range(len(vec))]
    distances = {}
    flag = None
    currentClusted = -1
    while len(clusterNodes) > n:
        minDistances = float("inf")
        lenClusterNodes = len(clusterNodes)
        for i in range(lenClusterNodes - 1):
            for j in range(i + 1, lenClusterNodes):
                if distances.get((clusterNodes[i].id, clusterNodes[j].id)) == None:
                    distances[(clusterNodes[i].id, clusterNodes[j].id)] = euclideanDistance(clusterNodes[i].vec,
                                                                                            clusterNodes[j].vec)
            d = distances[(clusterNodes[i].id, clusterNodes[j].id)]
            if d < minDistances:
                minDistances = d
                flag = (i, j)
        bic1, bic2 = flag
        newvec = [(clusterNodes[bic1].vec[i] + clusterNodes[bic2].vec[i]) / 2 for i in
                  range(len(clusterNodes[bic1].vec))]
        newbic = clusterNode(newvec, left=clusterNodes[bic1], right=clusterNodes[bic2], distance=minDistances,
                             id=currentClusted)  # 二合一
        currentClusted -= 1
        del clusterNodes[bic2]  # 删除聚成一起的两个数据，由于这两个数据要聚成一起
        del clusterNodes[bic1]
        clusterNodes.append(newbic)  # 补回新聚类中心
        clusters = [yezi(clusterNodes[i]) for i in range(len(clusterNodes))]  # 深度优先搜索叶子节点，用于输出显示
    return clusterNodes, clusters


c = [[123, 312, 434, 4325, 345345], [23124, 141241, 434234, 9837489, 34743], [128937, 127, 12381, 424, 8945],
     [323, 4348, 5040, 8189, 2348], [51249, 42190, 2713, 2319, 4328], [13957, 1871829, 8712847, 34589, 30945],
     [1234, 45094, 23409, 13495, 348052], [49853, 3847, 4728, 4059, 5389]]

k, l = hcluster(c, 4)
print(l)
