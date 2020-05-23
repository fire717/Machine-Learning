#coding:utf-8

import kNN 
import matplotlib
import matplotlib.pyplot as plt
from numpy import *

#group,labels = kNN.createDataSet()

#kNN.classify0([0,0], group,labels,3) 

datingDataMat,datingLabels = kNN.file2matrix('datingTestSet2.txt')
"""
fig = plt.figure()   #表示绘制一个图
ax = fig.add_subplot(111) #将画布分割成1行1列，图像画在从左到右从上到下的第1块
ax.scatter(datingDataMat[:,1],datingDataMat[:,2],15.0*array(datingLabels),15.0*array(datingLabels)) #scatter生成散点图函数。使用datingDataMat矩阵的第二、三列数据
                                                    #datingDataMat[:,1]意思是所有行的第2列(从0开始)
                                                    #后面第一个数字参数对应左边两种颜色的点的半径大小，第二个数字试了下没什么变化
plt.show()
"""

normMat,ranges,minVals = kNN.autoNorm(datingDataMat)

#print kNN.datingClassTest()


#print kNN.classifyPerson()

#testVector = kNN.img2vector('testDigits/0_13.txt')
#print testVector[0,32:63]

print kNN.handwritingClassTest()
