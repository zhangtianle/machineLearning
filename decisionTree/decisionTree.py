import csv
import pydotplus
from sklearn import tree
from sklearn import preprocessing
from sklearn.feature_extraction import DictVectorizer

featureList = []
labelList = []

with open('AllElectronics.csv') as f:
    f_csv = csv.reader(f)
    headers = next(f_csv)
    for row in f_csv:
        labelList.append(row[len(row)-1])
        rowDic = {}
        for col in range(1, len(row)-1):
            rowDic[headers[col]] = row[col]
        featureList.append(rowDic)

# Vetorize features
vec = DictVectorizer()
dummyX = vec.fit_transform(featureList) .toarray()

# vectorize class labels
lb = preprocessing.LabelBinarizer()
dummyY = lb.fit_transform(labelList)
# print("dummyY: " + str(dummyY))

# Using decision tree for classification
# clf = tree.DecisionTreeClassifier()
clf = tree.DecisionTreeClassifier(criterion='entropy')
clf = clf.fit(dummyX, dummyY)


# Visualize model
dot_data = tree.export_graphviz(clf, out_file=None)
graph = pydotplus.graph_from_dot_data(dot_data)
graph.write_pdf("iris.pdf")
