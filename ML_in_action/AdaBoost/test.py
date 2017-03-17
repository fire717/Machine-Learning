#coding:utf-8

import adaboost
from numpy import *

datMat,classLabels=adaboost.loadSimpData()

#print datMat,classLabels

D = mat(ones((5,1))/5)
#print adaboost.buildStump(datMat,classLabels,D)

classifierArray = adaboost.adaBoostTrainDS(datMat,classLabels,9)
print classifierArray
