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
    sqDiffMat = diffMat**2                  #每个inX里的每个维度分别平方
    sqDistances = sqDiffMat.sum(axis=1)    #对矩阵axis=0表示列，axis=1表示行
    distances = sqDistances**0.5           #分别对每个元素开方
    sortedDistIndicies = distances.argsort()   #argsort函数返回的是数组值从小到大的索引值，eg. x=np.array([3, 1, 2]) -> np.argsort(x) -> array([1, 2, 0])
    classCount={}
    for i in range(k):
        voteIlabel = labels[sortedDistIndicies[i]]  #按距离从小到大的顺序取前K个，然后取得其对应的标签值
        classCount[voteIlabel] = classCount.get(voteIlabel,0) + 1   #dict.get(key, default=None) 返回给定键的值。如果键不可用，则返回默认值None。
                                                                #这里的作用就是统计每个标签出现的次数
    sortedClassCount = sorted(classCount.iteritems(),key=operator.itemgetter(1),reverse=True)
    #sort()与sorted()的不同在于，sort是在原位重新排列列表，而sorted()是产生一个新的列表。
    #dict.iteritems() 迭代返回键值对
    #key=operator.itemgetter(1) 按照对象的第1个域的值排序
    #reverse=True 降序排序
    return sortedClassCount[0][0]   #打印出排好序的键值对(list)的第一对的第一个值
