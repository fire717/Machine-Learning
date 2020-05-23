#coding:utf-8
from numpy import *

#SMO(序列最小优化，用于训练SVM)算法中的辅助函数
def loadDataSet(fileName):    #加载数据
    dataMat = []; labelMat = []
    fr = open(fileName)
    for line in fr.readlines():
        lineArr = line.strip().split('\t')
        dataMat.append([float(lineArr[0]), float(lineArr[1])])
        labelMat.append(float(lineArr[2]))
    return dataMat,labelMat     #返回数据集和标签集

def selectJrand(i,m):   #i是alpha的下标，m是所有alpha（拉格朗日乘子）的数目 / 只要函数值不等于输入值i，函数就会进行随机选择
    j=i
    while  (j==i):
        j = int(random.uniform(0,m))
    return j        #返回不等于输入值得一个随机下标值

def clipAlpha(aj,H,L):   #调整大于H或者小于L的alpha的值
    if aj > H:
        aj = H
    if L > aj:
        aj = L
    return aj


#简化版SMO算法
def smoSimple(dataMatIn,classLabels,C,toler,maxIter):
    #参数：数据集、类别标签、常数C、容错率、退出前最大循环次数。
    dataMatrix = mat(dataMatIn); labelMat = mat(classLabels).transpose()#转置为列向量
    b = 0; m,n = shape(dataMatrix)
    alphas = mat(zeros((m,1)))
    iter = 0  #没有任何α改变的情况下遍历数据集的次数
    while (iter < maxIter):
        alphaPairsChanged = 0 #按对改变α的值，这里记录是否更改
        for i in range(m):
            fXi = float(multiply(alphas,labelMat).T*(dataMatrix*dataMatrix[i,:].T)) + b #由观察大胆猜测，.T是转置的意思
            #fXi是什么意思还在大胆猜测中...
            Ei = fXi - float(labelMat[i]) #这里相减 难道是预测标签的意思吗
            if ((labelMat[i]*Ei < -toler) and (alphas[i] < C)) or ((labelMat[i]*Ei > toler) and (alphas[i] > 0)):
                j = selectJrand(i,m)    #随机选择第二个α
                fXj = float(multiply(alphas,labelMat).T*(dataMatrix*dataMatrix[j,:].T)) + b
                Ej = fXj - float(labelMat[j])
                alphaIold = alphas[i].copy(); #拷贝一份  记录i，j两点原始的位置，便于后面计算修改量
                alphaJold = alphas[j].copy();

                if (labelMat[i] != labelMat[j]):  #保证α在0与C之间
                    L = max(0,alphas[j] - alphas[i])
                    H = min(C,C + alphas[j] - alphas[i])
                else:
                    L = max(0,alphas[j] + alphas[i] - C)
                    H = min(C,alphas[j] + alphas[i])
                if L==H:
                    print "L==H";continue
                eta = 2.0 * dataMatrix[i,:]*dataMatrix[j,:].T - dataMatrix[i,:]*dataMatrix[i,:].T - dataMatrix[j,:]*dataMatrix[j,:].T
                if eta >= 0:
                    print "eta>=0";continue
                alphas[j] -= labelMat[j]*(Ei - Ej)/eta
                alphas[j] = clipAlpha(alphas[j],H,L)
                if (abs(alphas[j] - alphaJold) < 0.00001): 
                    print "j not moving enough";continue
                alphas[i] += labelMat[j]*labelMat[i]*(alphaJold - alphas[j])  #对i进行修改，修改量与j相同但方向相反
                b1 = b -Ei - labelMat[i]*(alphas[i]-alphaIold)*dataMatrix[i,:]*dataMatrix[i,:].T - labelMat[j]*(alphas[j] - alphaJold)*dataMatrix[i,:]*dataMatrix[j,:].T
                b2 = b -Ej - labelMat[i]*(alphas[i]-alphaIold)*dataMatrix[i,:]*dataMatrix[j,:].T - labelMat[j]*(alphas[j] - alphaJold)*dataMatrix[j,:]*dataMatrix[j,:].T 
                if (0 < alphas[i]) and (C > alphas[j]):    #设置常数项
                    b = b1  
                elif (0 < alphas[j]) and (C > alphas[j]):
                    b = b2
                else:
                    b = (b1+b2)/2.0
                alphaPairsChanged += 1
                print "iter: %d i:%d, pairs changed %d" % (iter,i,alphaPairsChanged)
        if (alphaPairsChanged == 0):
            iter += 1
        else:
            iter = 0
        print "iteration number: %d" % iter
    return b,alphas
