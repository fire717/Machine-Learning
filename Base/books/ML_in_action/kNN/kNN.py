#coding:utf-8

from numpy import *
from os import listdir    #列出给定目录的文件名
import operator

def createDataSet():
    group = array([[1.0,1.1],[1.0,1.0],[0,0],[0,0.1]])
    labels = ['A','A','B','B']
    return group, labels

#用于对基本例子分类的函数
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

#用于将文本记录转换为NUMPY的解析程序
def file2matrix(filename):   
    fr = open(filename)
    arrayOLines = fr.readlines()        #一次读取整个文件，自动将文件内容分析成一个行的列表
    numberOfLines = len(arrayOLines)    #得到文件行数
    returnMat = zeros((numberOfLines,3))   #np中的zero()函数功能是创建给定类型的矩阵，并初始化为0 / 【要from numpy import * 才行，直接import不可以】
    classLabelVector = []
    index = 0
    for line in arrayOLines:
        line = line.strip()    #strip() 方法用于移除字符串头尾指定的字符（默认为空格）
        listFromLine = line.split('\t') #\t 代表的是制表符（相当于tab）,split()通过指定分隔符对字符串进行切片
        returnMat[index,:] = listFromLine[0:3] #选取前三个元素，存储在特征矩阵中，test了下[index,:] 只能针对zeros生成的矩阵，意思是把后面[0:3]全部通过[,:]提取然后存储在对应index的行里
        classLabelVector.append(int(listFromLine[-1])) #每一行的最后一个参数存储为类别
        index += 1
    return returnMat,classLabelVector


#归一化特征值
def autoNorm(dataSet):
    minVals = dataSet.min(0)    #0指取最小的一行
    maxVals = dataSet.max(0)
    ranges = maxVals - minVals
    normDataSet = zeros(shape(dataSet))   #生成一个与dataset同型的矩阵，以0填充
    m = dataSet.shape[0]    #dataset的行数
    normDataSet = dataSet -tile(minVals,(m,1))   #tile（A,reps），A：输入的array，reps：array_like A沿各个维度重复的次数
                            #reps的数字从后往前分别对应A的第N个维度的重复次数。tile（A,(2,3)）表示A的第一个维度（这里是一行）重复3遍，第二个维度（一列）重复2遍
    normDataSet = normDataSet/tile(ranges,(m,1)) #在np中的除法值每个元素分别相除而不是矩阵除法
    return normDataSet,ranges,minVals

#测试分类器效果函数
def datingClassTest():
    hoRatio = 0.10
    datingDataMat,datingLabels = file2matrix('datingTestSet2````````.txt')
    normMat,ranges,minVals = autoNorm(datingDataMat)
    m = normMat.shape[0]  #行数
    numTestVecs = int(m*hoRatio)
    errorCount = 0.0
    for i in range(numTestVecs):
        classifierResult = classify0(normMat[i,:],normMat[numTestVecs:m,:],datingLabels[numTestVecs:m],3)
        print "the classifier came back with: %d,the real answer is:%d" % (classifierResult,datingLabels[i])
        if (classifierResult != datingLabels[i]):errorCount+=1.0
    print "the total error rate is:%f" % (errorCount/float(numTestVecs))

#约会网站预测函数
def classifyPerson():
    resultList = ['not at all','in small doses','in large doses']
    percentTats = float(raw_input("percentage of time spent playing vedio games?"))
    ffMiles = float(raw_input("frequent flier miles earned per year?"))
    iceCream = float(raw_input("liters of ice cream consumed per year?"))
    datingDataMat,datingLabels = file2matrix('datingTestSet2.txt')
    normMat,ranges,minVals=autoNorm(datingDataMat)
    inArr = array([ffMiles,percentTats,iceCream])
    classifierResult = classify0((inArr-minVals)/ranges,normMat,datingLabels,3)
    print "You will probably like this person: ",resultList[classifierResult - 1]

#手写数字识别
#图像转换成向量
def img2vector(filename):
    returnVect = zeros((1,1024))  #看来一般都是先生成好矩阵，这样就不用后面慢慢一个个append了
    fr = open(filename)
    for i in range(32):   #前32行
        lineStr = fr.readline()
        for j in range(32):   #每行的前32列
            returnVect[0,32*i+j] = int(lineStr[j])  #第一个0相当于第0行吧
    return returnVect

#手写数字识别系统的测试代码
def handwritingClassTest():
    hwLables = []

    trainingFileList = listdir('trainingDigits')   #获取目录内容
    m = len(trainingFileList)     #目录下文件数量
    trainingMat = zeros((m,1024))
    for i in range(m):
        fileNameStr = trainingFileList[i]     #获取文件名
        fileStr = fileNameStr.split('.')[0]     #以.分割获取前半部分，即去掉.txt
        classNumStr = int(fileStr.split('_')[0]) #获取分类的数字
        hwLables.append(classNumStr)    #存储为对应的标签
        trainingMat[i,:] = img2vector('trainingDigits/%s' % fileNameStr)  #把每一行的文件都转为向量

    testFileList = listdir('testDigits')
    errorCount = 0.0
    mTest = len(testFileList)
    for i in range(mTest):
        fileNameStr =testFileList[i]
        fileStr = fileNameStr.split('.')[0]
        classNumStr = int(fileStr.split('_')[0])
        vectorUnderTest = img2vector('testDigits/%s' % fileNameStr)      #测试数据处理同上
        classifierResult = classify0(vectorUnderTest,trainingMat,hwLables,3)  #通过循环测试每一个文件，用到了整个traingMat
        print "the classifier came back with: %d,the real answer is: %d" % (classifierResult,classNumStr)
        if (classifierResult != classNumStr):errorCount+=1.0
    print "\nthe total number of errors is:%d" % errorCount
    print "\nthe total error rate is: %f" % (errorCount/float(mTest))
