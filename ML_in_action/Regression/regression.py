#coding:utf-8
from numpy import *

def loadDataSet(fileName):
    numFeat = len(open(fileName).readline().split('\t')) - 1
    dataMat = []; labelMat = []
    fr = open(fileName)
    for line in fr.readlines():
        lineArr = []
        curLine = line.strip().split('\t')
        for i in range(numFeat):
            lineArr.append(float(curLine[i]))
        dataMat.append(lineArr)
        labelMat.append(float(curLine[-1]))
    return dataMat,labelMat

#计算最佳拟合直线
def standRegres(xArr,yArr):
    xMat = mat(xArr);yMat =mat(yArr).T
    xTx = xMat.T*xMat
    if linalg.det(xTx) == 0.0:  #np提供的线性函数库linalg，其中linalg.det(x)函数计算行列式的值
        print 'this matrix is singular,cannot do inverse'
        return
    ws = xTx.I * (xMat.T*yMat) #.T是转置的话这里.I就是逆了呗  / 这里都是用的p138的公式
    return ws #预测的参数向量


#####局部加权线性回归函数 / 给定x空间中任意一点，计算出对应的预测值yHat
def lwlr(testPoint,xArr,yArr,k=1.0):
    xMat = mat(xArr); yMat = mat(yArr).T
    m = shape(xMat)[0]
    weights = mat(eye((m)))  #numpy.eye()创建对角矩阵
    for j in range(m):
        diffMat = testPoint - xMat[j,:]
        weights[j,j] = exp(diffMat*diffMat.T/(-2.0*k**2))  #权值大小以指数级衰减，参数k控制衰减速度
    xTx = xMat.T * (weights * xMat)
    if linalg.det(xTx) == 0.0:  #判断行列式是否为0
        print 'this matrix is singular, cannot do inverse'
        return
    ws = xTx.I * (xMat.T * (weights * yMat))   #按照书上的公式 / 得到对回归系数ws的一个估计
    return testPoint *ws

def lwlrTest(testArr,xArr,yArr,k=1.0):   #用于为数据集中每个点调用lwlr()，有助于求解k的大小
    m = shape(testArr)[0]
    yHat = zeros(m)
    for i in range(m):
        yHat[i] = lwlr(testArr[i],xArr,yArr,k)
    return yHat

#缩减系数：岭回归（在矩阵XT*X上加一个kI从而使矩阵非奇异，进而能求逆，I为单位矩阵 / 缩减法的一种，相当于对回归系数的大小施加了限制）
#lasso：限制回归系。难以求解，可使用计算简便的逐步线性回归法（属于贪心算法，每一步都尽可能减少误差）求近似结果。

#权衡方差与误差。可指出哪些特征时关键的，哪些是不重要的。
