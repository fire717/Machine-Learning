#coding:utf-8

import bayes

listOPosts,listClasses = bayes.loadDataSet()
myVocabList = bayes.createVocabList(listOPosts)  #构建一个包含所有词的词汇表
#print myVocabList

#print bayes.setOfWords2Vec(myVocabList,listOPosts[0])
#print bayes.setOfWords2Vec(myVocabList,listOPosts[3])


trainMat = []
for postinDoc in listOPosts:            #循环使用词向量来填充trainMat列表
    trainMat.append(bayes.setOfWords2Vec(myVocabList,postinDoc))   # 把训练样本的每一项文档中的词在词汇表中出现的位置标识成1，然后把所有词向量构成一个矩阵
p0V,p1V,pAb = bayes.trainNB0(trainMat,listClasses)   #概率向量

#print p0V,p1V
