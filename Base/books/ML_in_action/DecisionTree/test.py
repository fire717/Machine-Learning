#coding:utf-8
import trees
import treePlotter
import sys
reload(sys)
sys.setdefaultencoding('utf8')

#myDat[0][-1]='maybe'
#print trees.calcShannonEnt(myDat)
#print trees.chooseBestFeatureToSplit(myDat)
#print trees.splitDataSet(myDat,0,0)
#print trees.splitDataSet(myDat,0,1)

myDat,labels = trees.createDataSet()
#print myDat
#print trees.createTree(myDat,labels) 

#treePlotter.createPlot()

#print treePlotter.retrieveTree(1)

myTree = treePlotter.retrieveTree(0)
#print treePlotter.getNumLeafs(myTree),treePlotter.getTreeDepth(myTree)

#myTree['no surfacing'][3]='maybe'
#print myTree
#treePlotter.createPlot(myTree)

'''
print labels
print myTree
print trees.classify(myTree,labels,[1,0])
print trees.classify(myTree,labels,[1,1])
'''

#trees.storeTree(myTree,'classifierStorage.txt')
#print trees.grabTree('classifierStorage.txt')

#预测隐形眼镜类型
fr=open('lenses.txt')
lenses = [inst.strip().split('\t') for inst in fr.readlines()]
lensesLabels = ['age','prescript','astigmatic','tearRate']
lensesTree = trees.createTree(lenses,lensesLabels)
print lensesTree
treePlotter.createPlot(lensesTree)
