#coding:utf-8
#使用文本注解绘制树节点

import matplotlib.pyplot as plt

decisionNode = dict(boxstyle="sawtooth",fc="0.8")  #声明字典的一种新方式
leafNode = dict(boxstyle="round4",fc="0.8")
arrow_args = dict(arrowstyle="<-")      #定义文本框和箭头样式

#绘制带箭头的注解
def plotNode(nodeTxt,centerPt,parentPt,nodeType):
    createPlot.axl.annotate(nodeTxt,xy=parentPt,xycoords='axes fraction',\
                            xytext=centerPt,textcoords='axes fraction',\
                            va="center",ha="center",bbox=nodeType,arrowprops=arrow_args)

#在父子节点间填充文本信息
def plotMidText(cntrPt,parentPt,txtString):
    xMid = (parentPt[0]-cntrPt[0])/2.0 + cntrPt[0]
    yMid = (parentPt[1]-cntrPt[1])/2.0 + cntrPt[1]
    createPlot.axl.text(xMid,yMid,txtString)

#
def plotTree(myTree,parentPt,nodeTxt):
    numLeafs = getNumLeafs(myTree)
    depth = getTreeDepth(myTree)    #计算宽与高
    firstStr = myTree.keys()[0]
    cntrPt = (plotTree.xOff +(1.0+float(numLeafs))/2.0/plotTree.totalW,plotTree.yOff)  #全局变量plotTree.xOff/.yOff追踪已经绘制的节点位置 
    plotMidText(cntrPt,parentPt,nodeTxt)   #标记子节点属性值
    plotNode(firstStr,cntrPt,parentPt,decisionNode)
    secondDict = myTree[firstStr]
    plotTree.yOff = plotTree.yOff - 1.0/plotTree.totalD   #减少y偏移
    for key in secondDict.keys():
        if type(secondDict[key]).__name__=='dict':
            plotTree(secondDict[key],cntrPt,str(key))
        else:
            plotTree.xOff = plotTree.xOff +1.0/plotTree.totalW
            plotNode(secondDict[key],(plotTree.xOff,plotTree.yOff),cntrPt,leafNode)
            plotMidText((plotTree.xOff,plotTree.yOff),cntrPt,str(key))
    plotTree.yOff = plotTree.yOff + 1.0/plotTree.totalD


def createPlot(inTree):
    fig = plt.figure(1,facecolor='white')
    fig.clf()
    axprops = dict(xticks=[],yticks=[])
    createPlot.axl = plt.subplot(111,frameon=False,**axprops)
    plotTree.totalW = float(getNumLeafs(inTree))
    plotTree.totalD = float(getTreeDepth(inTree))
    plotTree.xOff = -0.5/plotTree.totalW; plotTree.yOff = 1.0;  #设置根节点在y轴1.0，x轴宽度的1/2处
    plotTree(inTree,(0.5,1.0),'')
    plt.show()

#获取叶节点的数目
def getNumLeafs(myTree):
    numLeafs = 0
    firstStr = myTree.keys()[0]  #dict.keys()返回字典中所有关键字组成的list
    secondDict = myTree[firstStr] #因为子树下都是从0,1开始，所以这里可以当做关键字 参看49行
    for key in secondDict.keys():
        if type(secondDict[key]).__name__=='dict':    #测试节点数据类型是否为字典,可通过__name__访问
            numLeafs += getNumLeafs(secondDict[key])  #递归
        else: numLeafs+=1
    return numLeafs

#获取树的层数
def getTreeDepth(myTree):
    maxDepth = 0
    firstStr = myTree.keys()[0]   
    secondDict = myTree[firstStr] 
    for key in secondDict.keys():
        if type(secondDict[key]).__name__=='dict':
            thisDepth = 1 + getTreeDepth(secondDict[key])
        else:thisDepth = 1
        if thisDepth > maxDepth:maxDepth = thisDepth
    return maxDepth

#节省时间，输出预先存储的树信息/ 主要用于测试
def retrieveTree(i):
    listOfTrees = [{'no surfacing':{0:'no',1:{'flippers':{0:'no',1:'yes'}}}},
                    {'no surfacing':{0:'no',1:{'flippers':{0:{'head':{0:'no',1:'yes'}},1:'no'}}}}
                    ]
    return listOfTrees[i]
