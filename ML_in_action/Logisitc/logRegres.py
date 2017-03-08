#coding:utf-8
from numpy import *

#####Logistic回归梯度上升优化算法#####
##读取数据集，生成数据点矩阵和标签矩阵
def loadDataSet():   
    dataMat = []; labelMat = []
    fr = open('testSet.txt')
    for line in fr.readlines():  #按行读取
        lineArr = line.strip().split()  #去掉首尾空格再以空格分割
        dataMat.append([1.0, float(lineArr[0]), float(lineArr[1])])
        #每个回归系数初始化为1 ，注意这里的float
        labelMat.append(int(lineArr[2]))
    return dataMat,labelMat

def sigmoid(inX):
    return 1.0/(1+exp(-inX))  #exp()即e的多少次方

def gradAscent(dataMatIn, classLabels):
    dataMatrix = mat(dataMatIn)             #convert to NumPy matrix
    labelMat = mat(classLabels).transpose() #transpose() 矩阵转置成m行1列
    m,n = shape(dataMatrix)   #shape返回两个值，行和列
    alpha = 0.001           #向目标移动的步长
    maxCycles = 500
    weights = ones((n,1)) #生成n行1列的矩阵，以1填充
    for k in range(maxCycles):              
        h = sigmoid(dataMatrix*weights)     #weight是用来迭代的，所以初始全1.矩阵乘法，n列*n行，结果h为m行1列
        error = (labelMat - h)             #h是0-1的，则error有正有负，对应w也会+-。即当error为负时，说明h比标签大，所以w需要变小，所以就是-
        weights = weights + alpha * dataMatrix.transpose()* error #error 为m行1列，所以数据要转置
                    #当h预测对时，与标签相同差为0，则w+0=w不变，而有差异时，差异越大w+的越多，所以能整体上同时逼近
                    #为了稳定，限制了步长
    return weights
################################################
