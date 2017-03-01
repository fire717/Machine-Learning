#coding:utf-8

import kNN 

group,labels = kNN.createDataSet()

print kNN.classify0([0,0], group,labels,3)
