#coding:utf-8

from numpy import *

################K-means聚类支持函数
def loadDataSet(fileName):
    dataMat = []
    fr = open(fileName)
    for line in fr.readlines():
        curLine = line.strip().split('\t')
        fltLine = map(float,curLine)  #每一行float
        dataMat.append(fltLine)   #把一行当做一个整体list append到mat中
    return dataMat

#计算两个向量的欧式距离
def distEclud(vecA,vecB):
    return sqrt(sum(power(vecA-vecB,2)))   #sqrt()求平方根 / 这里是每一维对应相减再平方
                                           #numpy.power(x1, x2)对x1数组的元素分别求x2次方。x2可以是数字，也可以是数组，但是x1和x2的列数要相同。

#为给定数据集构建一个包含k个随机质心的集合
def randCent(dataSet,k):
    n = shape(dataSet)[1]   #获取数据集的列数，即特征数
    centroids = mat(zeros((k,n)))  #生成一个质心矩阵，k行n列  /  zeros生成二维矩阵就要两个括号
    for j in range(n):   #按列处理
        minJ = min(dataSet[:,j])   #代表每一行的最小值
        rangeJ = float(max(dataSet[:,j]) - minJ)  #代表每一行的上下限范围
        centroids[:,j] = minJ + rangeJ * random.rand(k,1) #random.rand(k,1)生成k行1列的随机数（范围在0-1）   /  计算后赋值给cent的第j列
    return centroids
####################################


###############K-means均值聚类算法
def kMeans(dataSet,k,distMeas=distEclud,createCent=randCent):  
    #distMeas求距离的函数 和 createCent创建初始质心的函数 都是可选的
    m = shape(dataSet)[0]     #数据集中数据点的总数
    clusterAssment = mat(zeros((m,2)))  #簇分配结果矩阵为两列，第一列记录簇的索引值，第二列记录误差（当前点到簇质心的距离）
    centroids =createCent(dataSet,k)
    clusterChanged = True
    while clusterChanged:  #迭代，到所有数据点的簇分配结果不变为止
        clusterChanged = False
        for i in range(m):  #为每个点找到距离最近的簇中心
            minDist = inf;minIndex = -1    #inf指无穷大，赋初值比较好
            for j in range(k): #遍历每一个质心   
                distJI = distMeas(centroids[j,:],dataSet[i,:])
                if distJI < minDist:
                    minDist = distJI;minIndex = j
            if clusterAssment[i,0] != minIndex:
                clusterChanged = True   #如果索引值有变化，则继续迭代
            clusterAssment[i,:] = minIndex,minDist**2  #距离平方下保证非负
        print centroids
        for cent in range(k):   #更新所有质心的位置
            ptsInclust = dataSet[nonzero(clusterAssment[:,0].A==cent)[0]] #通过数组过滤获得给定簇的所有点
                        #clusterAssment[:,0].A==cent cA矩阵的第一列（即簇的索引值）   / 取第一列等于cent的所有列
                        #nonzeros(a)返回的是矩阵中所有非零元素的坐标(这里再取[0]代表行数)
            centroids[cent,:] = mean(ptsInclust,axis=0) #计算所有点的均值，axis=0意思是沿矩阵【列】方向进行均值计算
    return centroids,clusterAssment 


###############【二分K-means均值聚类算法】###########
def biKmeans(dataSet, k, distMeas=distEclud):
    m = shape(dataSet)[0]  #行数
    clusterAssment = mat(zeros((m,2)))
    centroid0 = mean(dataSet, axis=0).tolist()[0]   #创建一个初始簇。取整体数据集的列平均值。tolist转换为list对象。然后取第一列
    centList =[centroid0] #把第一列的均值再存入一个簇的list
    for j in range(m):#计算初始的距离差值
        clusterAssment[j,1] = distMeas(mat(centroid0), dataSet[j,:])**2
    while (len(centList) < k): #当划分的簇等于k时停止划分
        lowestSSE = inf #无穷大初值
        for i in range(len(centList)):
            ptsInCurrCluster = dataSet[nonzero(clusterAssment[:,0].A==i)[0],:]#获取所有属于当前簇的所有数据
            centroidMat, splitClustAss = kMeans(ptsInCurrCluster, 2, distMeas) #划分当前簇
            sseSplit = sum(splitClustAss[:,1])  #计算划分后的两个簇的SSE误差平方和
            sseNotSplit = sum(clusterAssment[nonzero(clusterAssment[:,0].A!=i)[0],1])
            print "sseSplit, and notSplit: ",sseSplit,sseNotSplit
            if (sseSplit + sseNotSplit) < lowestSSE:
                bestCentToSplit = i
                bestNewCents = centroidMat
                bestClustAss = splitClustAss.copy()   # copy()方法返回字典的浅拷贝
                lowestSSE = sseSplit + sseNotSplit
        bestClustAss[nonzero(bestClustAss[:,0].A == 1)[0],0] = len(centList) #change 1 to 3,4, or whatever
        bestClustAss[nonzero(bestClustAss[:,0].A == 0)[0],0] = bestCentToSplit
        print 'the bestCentToSplit is: ',bestCentToSplit
        print 'the len of bestClustAss is: ', len(bestClustAss)
        centList[bestCentToSplit] = bestNewCents[0,:].tolist()[0]#replace a centroid with two best centroids 
        centList.append(bestNewCents[1,:].tolist()[0])
        clusterAssment[nonzero(clusterAssment[:,0].A == bestCentToSplit)[0],:]= bestClustAss#reassign new clusters, and SSE
    return mat(centList), clusterAssment


######对地图上的点进行聚类/ 对地理数据应用二分K-均值算法
import urllib
import json
def geoGrab(stAddress, city):
    apiStem = 'http://where.yahooapis.com/geocode?'  #create a dict and constants for the goecoder
    params = {}
    params['flags'] = 'J'#JSON return type
    params['appid'] = 'aaa0VN6k'
    params['location'] = '%s %s' % (stAddress, city)
    url_params = urllib.urlencode(params)
    yahooApi = apiStem + url_params      #print url_params
    print yahooApi
    c=urllib.urlopen(yahooApi)
    return json.loads(c.read())

from time import sleep
def massPlaceFind(fileName):
    fw = open('places.txt', 'w')
    for line in open(fileName).readlines():
        line = line.strip()
        lineArr = line.split('\t')
        retDict = geoGrab(lineArr[1], lineArr[2])
        if retDict['ResultSet']['Error'] == 0:
            lat = float(retDict['ResultSet']['Results'][0]['latitude'])
            lng = float(retDict['ResultSet']['Results'][0]['longitude'])
            print "%s\t%f\t%f" % (lineArr[0], lat, lng)
            fw.write('%s\t%f\t%f\n' % (line, lat, lng))
        else: print "error fetching"
        sleep(1)
    fw.close()


#####球面距离计算及簇绘图函数    
def distSLC(vecA, vecB):#Spherical Law of Cosines
    a = sin(vecA[0,1]*pi/180) * sin(vecB[0,1]*pi/180)
    b = cos(vecA[0,1]*pi/180) * cos(vecB[0,1]*pi/180) * \
                      cos(pi * (vecB[0,0]-vecA[0,0]) /180)
    return arccos(a + b)*6371.0 #pi is imported with numpy

import matplotlib
import matplotlib.pyplot as plt
def clusterClubs(numClust=5):
    datList = []
    for line in open('places.txt').readlines():
        lineArr = line.split('\t')
        datList.append([float(lineArr[4]), float(lineArr[3])])
    datMat = mat(datList)
    myCentroids, clustAssing = biKmeans(datMat, numClust, distMeas=distSLC)
    fig = plt.figure()
    rect=[0.1,0.1,0.8,0.8]
    scatterMarkers=['s', 'o', '^', '8', 'p', \
                    'd', 'v', 'h', '>', '<']
    axprops = dict(xticks=[], yticks=[])
    ax0=fig.add_axes(rect, label='ax0', **axprops)
    imgP = plt.imread('Portland.png')
    ax0.imshow(imgP)            #基于图像创建矩阵
    ax1=fig.add_axes(rect, label='ax1', frameon=False)      
    for i in range(numClust):
        ptsInCurrCluster = datMat[nonzero(clustAssing[:,0].A==i)[0],:]
        markerStyle = scatterMarkers[i % len(scatterMarkers)]
        ax1.scatter(ptsInCurrCluster[:,0].flatten().A[0], ptsInCurrCluster[:,1].flatten().A[0], marker=markerStyle, s=90)
    ax1.scatter(myCentroids[:,0].flatten().A[0], myCentroids[:,1].flatten().A[0], marker='+', s=300)
    plt.show()
