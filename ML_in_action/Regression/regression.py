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
