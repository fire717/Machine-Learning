#coding:utf-8
import logRegres
from numpy import *


dataArr,labelMat=logRegres.loadDataSet()
#print logRegres.gradAscent(dataArr,labelMat)

#weights = logRegres.gradAscent(dataArr,labelMat)
#print weights,weights.getA()
#logRegres.plotBestFit(weights.getA())  #矩阵通过这个getA()这个方法可以将自身返回成一个n维数组对象,
                            #不然直接使用weights在plotBestFit函数中的weights[1]就不是一个数而是[ 0.48007329]了

weights = logRegres.stocGradAscent1(array(dataArr),labelMat)
logRegres.plotBestFit(weights)
