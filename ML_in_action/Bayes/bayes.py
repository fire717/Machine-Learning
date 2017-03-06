#coding:utf-8

from numpy import *

###词表到向量的转换函数###
#创建一些实验样本
def loadDataSet():
    postingList = [['my','dog','has','flea',
                    'problem','help','please'],
                   ['maybe','not','take','him',
                    'to','dog','park','stupid'],
                   ['my','dalmation','is','so','cute',
                    'I','love','him'],
                   ['stop','posting','stupid','worthless','garbage'],
                   ['mr','licks','ate','my','steak','how',
                    'to','stop','him'],
                   ['quit','buying','worthless','dog','food','stupid']]
    classVec = [0,1,0,1,0,1]  #1代表侮辱性文字，0代表正常
    return postingList,classVec

#创建一个包含在所有文档中出现的不重复词的列表
def createVocabList(dataSet):
    vocabSet = set([])    #创建一个空的集合类型
    for document in dataSet:
        vocabSet = vocabSet | set(document)  # | 按位或运算符, 这里用于求两个集合(只能是set类型)的【并集】           
    return list(vocabSet)

#判断词汇表中的词是否在输入文档中出现
def setOfWords2Vec(vocabList,inputSet):
    returnVec = [0]*len(vocabList)
    for word in inputSet:
        if word in vocabList:
            returnVec[vocabList.index(word)] = 1    #list.index(key)可以获取关键字的索引号   【在词汇表中出现了就设置该行对应的词向量的对应位置为1】
        else: print "the word: %s is not in my Vocabulary!" % word
    return returnVec       #返回的是词向量

#朴素贝叶斯分类器训练函数
def trainNB0(trainMatrix,trainCategory):
    #trainMatrix不是文档了，而是每一项文档中的词在词汇表中出现位置构成的0,1的矩阵
    numTrainDocs = len(trainMatrix)  #训练数据条数
    numWords = len(trainMatrix[0])          #词汇表共有多少词
    pAbusive = sum(trainCategory)/float(numTrainDocs)       #初始化概率 侮辱性数据占总的训练数据项的比例，这里sum(trainCategory)即把1求和
    p0Num = ones(numWords); p1Num = ones(numWords)  #给每一条数据生成一个初始向量矩阵
    p0Denom = 2.0; p1Denow = 2.0                       #为避免概率相乘时出现0，所以把分子都初始化为1，分母都初始化为2
    for i in range(numTrainDocs):
        if trainCategory[i] == 1:                   #如果是侮辱性词汇
            p1Num += trainMatrix[i]                 #向量相加,p1Num中全部为0，trainMatrix有0有1 /  作用在于统计那些位置的词在是否侮辱性词句中起到作用
            p1Denow += sum(trainMatrix[i])          #有0有1所以可以求和  作用在于统计该条侮辱性词句中有多少个词
        else:                                       #因为贝叶斯方法要比较两类概率大小，所以都要求
            p0Num += trainMatrix[i]
            p0Denom += sum(trainMatrix[i])
    p1Vect = log(p1Num/p1Denow)          #  对每个元素做除法  除以词数得到所占比例        
    p0Vect = log(p0Num/p0Denom)          #  由直接相处改为log对数，可以消除很多太小的数相乘造成的下溢出的问题
    return p0Vect,p1Vect,pAbusive

###朴素贝叶斯分类函数###
def classifyNB(vec2Classify,p0Vec,p1Vec,pClass1):
    p1 = sum(vec2Classify * p1Vec) + log(pClass1)
    p0 = sum(vec2Classify * p0Vec) + log(1.0 - pClass1)
    if p1 > p0:
        return 1
    else:
        return 0 

def testingNB():
    listOPosts,listClasses = loadDataSet()
    myVocabList = createVocabList(listOPosts)
    trainMat = []
    for postinDoc in listOPosts:
        trainMat.append(setOfWords2Vec(myVocabList,postinDoc))
    p0V,p1V,pAb = trainNB0(array(trainMat),array(listClasses))

    testEntry = ['love','my','dalmation']
    thisDoc = array(setOfWords2Vec(myVocabList,testEntry))
    print testEntry,'classified as:',classifyNB(thisDoc,p0V,p1V,pAb)

    testEntry = ['stupid','garbage']
    thisDoc = array(setOfWords2Vec(myVocabList,testEntry))
    print testEntry,'classified as:',classifyNB(thisDoc,p0V,p1V,pAb)
