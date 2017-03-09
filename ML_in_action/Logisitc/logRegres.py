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
    weights = ones((n,1)) #生成n行1列的矩阵，以1填充 / 这里的意思是每一项数据集乘以一个权值向量可以得到标签值，这里通过先假设全1，然后利用训练数据不断逼近真实值
    for k in range(maxCycles):              
        h = sigmoid(dataMatrix*weights)     #weight是用来迭代的，所以初始全1.矩阵乘法，n列*n行，结果h为m行1列
        error = (labelMat - h)             #h是0-1的，则error有正有负，对应w也会+-。即当error为负时，说明h比标签大，所以w需要变小，所以就是-
        weights = weights + alpha * dataMatrix.transpose()* error #error 为m行1列，所以数据要转置 / 这里是梯度上升所以是+，下降则为-,但是为啥我把上面改成h-l，这里换成-，结果还是一样的呢..
                    #当h预测对时，与标签相同差为0，则w+0=w不变，而有差异时，差异越大w+的越多，所以能整体上同时逼近
                    #为了稳定，限制了步长
    return weights
################################################

#画出数据集和Logisitic回归最佳拟合直线
def plotBestFit(weights):
    import matplotlib.pyplot as plt
    dataMat,labelMat=loadDataSet()
    dataArr = array(dataMat)
    n = shape(dataArr)[0]     #行数
    xcord1 = []; ycord1 = []
    xcord2 = []; ycord2 = []
    for i in range(n):
        if int(labelMat[i])== 1:                #属于1类就给xy1赋值
            xcord1.append(dataArr[i,1]); ycord1.append(dataArr[i,2])
        else:
            xcord2.append(dataArr[i,1]); ycord2.append(dataArr[i,2])
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(xcord1, ycord1, s=30, c='red', marker='s')
    ax.scatter(xcord2, ycord2, s=30, c='green')
    x = arange(-3.0, 3.0, 0.1)  #来自numpy，生成一个等差数列的数组，从-3开始到3结束，每次间隔为0.1
    y = (-weights[0]-weights[1]*x)/weights[2] #最佳拟合直线  由0=w0x0+w1x1+w2x2推导，这里的y就是x2，x0=1
    ax.plot(x, y)
    plt.xlabel('X1'); plt.ylabel('X2');
    plt.show()

#随机梯度上升
def stocGradAscent0(dataMatrix, classLabels):
    m,n = shape(dataMatrix)
    alpha = 0.01
    weights = ones(n)   #initialize to all ones
    for i in range(m):
        h = sigmoid(sum(dataMatrix[i]*weights))   #h和error都是数值，不是向量。
        error = classLabels[i] - h
        weights = weights + alpha * error * dataMatrix[i]
    return weights

#改进的随机梯度上升
def stocGradAscent1(dataMatrix, classLabels, numIter=150):
    m,n = shape(dataMatrix)
    weights = ones(n)   #initialize to all ones
    for j in range(numIter):
        dataIndex = range(m)
        for i in range(m):
            alpha = 4/(1.0+j+i)+0.0001    #alpha每次迭代式都需要调整 ,i j越大，移动距离就越小，初始值0.0001保证不会接近0
            randIndex = random.randint(0,len(dataIndex))#随机选取更新回归系数，可减少周期性的波动
            h = sigmoid(sum(dataMatrix[randIndex]*weights))
            error = classLabels[randIndex] - h
            weights = weights + alpha * error * dataMatrix[randIndex]
            del(dataIndex[randIndex])
    return weights
