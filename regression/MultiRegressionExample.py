from sklearn import linear_model
from numpy import genfromtxt

datapath = "./Delivery_Dummy.csv"
data = genfromtxt(datapath, delimiter=',')

x = data[:, :-1]
y = data[:, -1]

print(x)

mlr = linear_model.LinearRegression()
mlr.fit(x, y)

print(mlr)
print("coef:" + repr(mlr.coef_) + " intercept" + repr(mlr.intercept_))

xPredict = [90, 2, 0, 0, 1]
yPredict = mlr.predict(xPredict)

print("predict:" + repr(yPredict))
