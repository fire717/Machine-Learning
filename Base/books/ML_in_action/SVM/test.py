#coding:utf-8

import svm

dataArr,labelArr = svm.loadDataSet('testSet.txt')

#print labelArr

b,alphas = svm.smoSimple(dataArr,labelArr,0.6,0.001,40)

print b
print alphas[alphas>0]
