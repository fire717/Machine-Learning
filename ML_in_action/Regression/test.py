#coding:utf-8
from numpy import *
import regression

xArr,yArr=regression.loadDataSet('ex0.txt')
print xArr[0:2]

ws = regression.standRegres(xArr,yArr)
print ws

#使用新的ws值计算预测的值yHat
xMat = mat(xArr)
yMat = mat(yArr)
yHat = xMat*ws

#绘出数据集散点图和最佳拟合直线图
import matplotlib.pyplot as plt
fig = plt.figure()
ax = fig.add_subplot(111)
ax.scatter(xMat[:,1].flatten().A[0],yMat.T[:,0].flatten().A[0])

#为了绘制计算出的最佳拟合曲线，需要绘出yHat的值
#若直线上的数据点次序混乱，绘图时将会出现问题，固要先将点按照升序排列
xCopy = xMat.copy()
xCopy.sort(0)  #这个应该是np中的sort，意思是按照0维度排序
yHat = xCopy*ws
ax.plot(xCopy[:,1],yHat)
plt.show()
