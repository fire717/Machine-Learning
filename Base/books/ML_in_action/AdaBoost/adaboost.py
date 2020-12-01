#coding:utf-8
from numpy import *

#创建简单数据集
def loadSimpData():
    dataMat = matrix([[1.,2.1],
                    [2. ,1.1],
                    [1.3,1. ],
                    [1. ,1. ],
                    [2. ,1. ]])
    classLabels = [1.0,1.0,-1.0,-1.0,1.0]
    return dataMat,classLabels

#####树桩（单层决策树）分类器
#通过阈值比较对数据进行分类
def stumpClassify(dataMatrix,dimen,threshVal,threshIneq):
    #threshVal阈值    threshIneq不等的类型
    retArray = ones((shape(dataMatrix)[0],1)) #先全部设成1
    if threshIneq == 'lt': #gt 大于greater than/  lt小于 less than
        retArray[dataMatrix[:,dimen] <= threshVal] = -1.0  #数组过滤 / 若是大于，则把第dimen维上小于阈值的设为-1
    else:
        retArray[dataMatrix[:,dimen] > threshVal] = -1.0   # / 若是小于，则把第dimen维上大于阈值的设为-1
    return retArray

#遍历上面函数的所有可能输入值，找到数据集上最佳的单层决策树
def buildStump(dataArr,classLabels,D):
    #D权重向量
    dataMatrix = mat(dataArr);labelMat = mat(classLabels).T
    m,n = shape(dataMatrix)
    numSteps = 10.0 #用于在特征的所有可能值上进行遍历
    bestStump = {} # 存储给定权重向量D时所得到得最佳单层决策树的相关信息
    bestClasEst = mat(zeros((m,1)))
    minError = inf #先初始化为无穷大，用于寻找可能的最小错误率
    for i in range(n): #在数据集的所有特征上遍历
        rangeMin = dataMatrix[:,i].min();rangeMax = dataMatrix[:,i].max();  #这种极值求法应该是np中的，得到第i个特征(第i列)上的极值
        stepSize = (rangeMax-rangeMin)/numSteps #得到步长
        for j in range(-1,int(numSteps)+1): #在这些步上遍历
            for inequal in ['lt','gt']:  #在大于小于中切换不等式   / 这两层循环其实就是遍历了所有阈值取值的情况，且每种阈值取值对应两种情况：大于它为1还是小于为1
                threshVal = (rangeMin + float(j)*stepSize)
                predictedVals = stumpClassify(dataMatrix,i,threshVal,inequal) #分类预测结果
                errArr = mat(ones((m,1)))
                errArr[predictedVals == labelMat] = 0 #数组过滤得到误差向量
                weightedError = D.T*errArr   #计算加权错误率
                print "split:dim %d,thresh %.2f,thresh ineqal: %s,the weighted error is %.3f" % (i,threshVal,inequal,weightedError)
                if weightedError < minError:  #如果误差向量变小了，则在bestStump字典中保存该单层决策树
                    minError = weightedError
                    bestClasEst = predictedVals.copy()
                    bestStump['dim'] = i
                    bestStump['thresh'] = threshVal
                    bestStump['ineq'] = inequal
    return bestStump,minError,bestClasEst


#完整AdaBoost算法
#基于单层决策树的AdaBoost训练过程
def adaBoostTrainDS(dataArr,classLabels,numIt=40):
    #numIt迭代次数
    weakClassArr = []
    m = shape(dataArr)[0]
    D = mat(ones((m,1))/m)
    aggClassEst = mat(zeros((m,1)))
    for i in range(numIt):
        bestStump,error,classEst = buildStump(dataArr,classLabels,D)
        print "D:",D.T
        alpha = float(0.5*log((1.0-error)/max(error,1e-16)))
        bestStump['alpha'] = alpha
        weakClassArr.append(bestStump)
        print "classEst:",classEst.T
        expon = multiply(-1*alpha*mat(classLabels).T,classEst) #为下一次迭代计算D
        D = multiply(D,exp(expon))
        D = D/D.sum()    #D包含了每个数据点的权重
        aggClassEst += alpha*classEst     #错误率累加计算
        print "aggClassEst: ",aggClassEst
        aggErrors = multiply(sign(aggClassEst) != mat(classLabels).T,ones((m,1)))
        errorRate = aggErrors.sum() / m
        print "total error:",errorRate,'\n'
        if errorRate == 0.0:break
    return weakClassArr
