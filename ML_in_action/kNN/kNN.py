#coding:utf-8

from numpy import *
import operator

def createDataSet():
    group = array([[1.0,1.1],[1.0,1.0],[0,0],[0,0.1]])
    labels = ['A','A','B','B']
    return group, labels

def classify0(inX, dataSet, labels, k):
    #inx:用于分类的输入向量
    dataSetSize = dataSet.shape[0]   #获取有多少组数据。这里的shape是[4,2]，即4组，每组为2维

    diffMat = tile(inX, (dataSetSize,1)) - dataSet  #tile（A,reps） A沿各个维度重复的次数，reps的数字从右往左分别对应A的第N个维度的重复次数
                                            #eg. tile(inX,(4,1))=[inX,inX,inX,inX]
    sqDiffMat = diffMat**2     #每个inX里的每个维度分别平方
    sqDistances = sqDiffMat.sum(axis=1)
    distances = sqDistances**0.5

    sortedDistIndicies = distances.argsort()
    classCount={}
    for i in range(k):
        voteIlabel = labels[sortedDistIndicies[i]]
        classCount[voteIlabel] = classCount.get(voteIlabel,0) + 1

    sortedClassCount = sorted(classCount.iteritems(),
        key=operator.itemgetter(1),reverse=True)
    return sortedClassCount[0][0]
