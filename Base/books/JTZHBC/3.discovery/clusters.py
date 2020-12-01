#coding:utf-8
from PIL import Image,ImageDraw

def readfile(filename):
  lines=[line for line in file(filename)]
  
  # First line is the column titles
  colnames=lines[0].strip().split('\t')[1:]
  rownames=[]
  data=[]
  for line in lines[1:]:
    p=line.strip().split('\t')
    # First column in each row is the rowname
    rownames.append(p[0])
    # The data for this row is the remainder of the row
    data.append([float(x) for x in p[1:]])
  return rownames,colnames,data


from math import sqrt

def pearson(v1,v2):
  # Simple sums
  sum1=sum(v1)
  sum2=sum(v2)
  
  # Sums of the squares
  sum1Sq=sum([pow(v,2) for v in v1])
  sum2Sq=sum([pow(v,2) for v in v2])	
  
  # Sum of the products
  pSum=sum([v1[i]*v2[i] for i in range(len(v1))])
  
  # Calculate r (Pearson score)
  num=pSum-(sum1*sum2/len(v1))
  den=sqrt((sum1Sq-pow(sum1,2)/len(v1))*(sum2Sq-pow(sum2,2)/len(v1)))
  if den==0: return 0

  return 1.0-num/den

class bicluster: #代表聚类这个类型 / 分级聚类 采取树状图
  def __init__(self,vec,left=None,right=None,distance=0.0,id=None):
    self.left=left
    self.right=right
    self.vec=vec
    self.id=id
    self.distance=distance

def hcluster(rows,distance=pearson):
  distances={}
  currentclustid=-1

  # Clusters are initially just the rows 最开始的聚类就是数据集中的行
  clust=[bicluster(rows[i],id=i) for i in range(len(rows))]

  while len(clust)>1:
    lowestpair=(0,1)
    closest=distance(clust[0].vec,clust[1].vec)

    # loop through every pair looking for the smallest distance遍历每一个配对，找到最小距离
    for i in range(len(clust)):
      for j in range(i+1,len(clust)):
        # distances is the cache of distance calculations 缓存距离的计算值
        if (clust[i].id,clust[j].id) not in distances:  #保存每一对避免重复计算
          distances[(clust[i].id,clust[j].id)]=distance(clust[i].vec,clust[j].vec)

        d=distances[(clust[i].id,clust[j].id)]

        if d<closest:
          closest=d
          lowestpair=(i,j)

    # calculate the average of the two clusters 计算两个聚类的平均值
    mergevec=[
    (clust[lowestpair[0]].vec[i]+clust[lowestpair[1]].vec[i])/2.0 
    for i in range(len(clust[0].vec))]

    # create the new cluster
    newcluster=bicluster(mergevec,left=clust[lowestpair[0]],
                         right=clust[lowestpair[1]],
                         distance=closest,id=currentclustid)

    # cluster ids that weren't in the original set are negative不在原始集合中的聚类其id为负数
    currentclustid-=1
    del clust[lowestpair[1]]
    del clust[lowestpair[0]]
    clust.append(newcluster)

  return clust[0]

#以层级形式打印聚类树
def printclust(clust,labels=None,n=0):
  # indent to make a hierarchy layout
  for i in range(n): print ' ',
  if clust.id<0:
    # negative id means that this is branch负数标记代表分支
    print '-'
  else:
    # positive id means that this is an endpoint正数标记代表叶节点
    if labels==None: print clust.id
    else: print labels[clust.id]

  # now print the right and left branches
  if clust.left!=None: printclust(clust.left,labels=labels,n=n+1)
  if clust.right!=None: printclust(clust.right,labels=labels,n=n+1)

###以图像形式绘制树状图
def getheight(clust):
  # Is this an endpoint? Then the height is just 1 是叶节点则高度为1
  if clust.left==None and clust.right==None: return 1

  # Otherwise the height is the same of the heights of
  # each branch  否则高度为分支高度之和
  return getheight(clust.left)+getheight(clust.right)

#还需要知道根节点的总体误差
def getdepth(clust):
  # The distance of an endpoint is 0.0 叶节点距离为0
  if clust.left==None and clust.right==None: return 0

  # The distance of a branch is the greater of its two sides
  # plus its own distance枝节点的距离等于左右分支中较大者加上该枝节点自身距离
  return max(getdepth(clust.left),getdepth(clust.right))+clust.distance

#为每一个生成的聚类产生一个图片
def drawdendrogram(clust,labels,jpeg='clusters.jpg'):
  # height and width
  h=getheight(clust)*20
  w=1200
  depth=getdepth(clust)

  # width is fixed, so scale distances accordingly 宽度固定，所以需要对距离值调整
  # 缩放因子
  scaling=float(w-150)/depth

  # Create a new image with a white background
  img=Image.new('RGB',(w,h),(255,255,255))
  draw=ImageDraw.Draw(img)

  draw.line((0,h/2,10,h/2),fill=(255,0,0))    

  # Draw the first node
  drawnode(draw,clust,10,(h/2),scaling,labels)
  img.save(jpeg,'JPEG')

#画节点
def drawnode(draw,clust,x,y,scaling,labels):
  if clust.id<0:
    h1=getheight(clust.left)*20
    h2=getheight(clust.right)*20
    top=y-(h1+h2)/2
    bottom=y+(h1+h2)/2
    # Line length
    ll=clust.distance*scaling
    # Vertical line from this cluster to children    
    draw.line((x,top+h1/2,x,bottom-h2/2),fill=(255,0,0))    
    
    # Horizontal line to left item
    draw.line((x,top+h1/2,x+ll,top+h1/2),fill=(255,0,0))    

    # Horizontal line to right item
    draw.line((x,bottom-h2/2,x+ll,bottom-h2/2),fill=(255,0,0))        

    # Call the function to draw the left and right nodes    
    drawnode(draw,clust.left,x+ll,top+h1/2,scaling,labels)
    drawnode(draw,clust.right,x+ll,bottom-h2/2,scaling,labels)
  else:   
    # If this is an endpoint, draw the item label
    draw.text((x+5,y-7),labels[clust.id],(0,0,0))

###列聚类
def rotatematrix(data):
  newdata=[]
  for i in range(len(data[0])):
    newrow=[data[j][i] for j in range(len(data))]
    newdata.append(newrow)
  return newdata

import random

###k均值聚类
def kcluster(rows,distance=pearson,k=4):
  # Determine the minimum and maximum values for each point
  ranges=[(min([row[i] for row in rows]),max([row[i] for row in rows])) 
  for i in range(len(rows[0]))]

  # Create k randomly placed centroids
  clusters=[[random.random()*(ranges[i][1]-ranges[i][0])+ranges[i][0] 
  for i in range(len(rows[0]))] for j in range(k)]
  
  lastmatches=None
  for t in range(100):
    print 'Iteration %d' % t
    bestmatches=[[] for i in range(k)]
    
    # Find which centroid is the closest for each row
    for j in range(len(rows)):
      row=rows[j]
      bestmatch=0
      for i in range(k):
        d=distance(clusters[i],row)
        if d<distance(clusters[bestmatch],row): bestmatch=i
      bestmatches[bestmatch].append(j)

    # If the results are the same as last time, this is complete
    if bestmatches==lastmatches: break
    lastmatches=bestmatches
    
    # Move the centroids to the average of their members
    for i in range(k):
      avgs=[0.0]*len(rows[0])
      if len(bestmatches[i])>0:
        for rowid in bestmatches[i]:
          for m in range(len(rows[rowid])):
            avgs[m]+=rows[rowid][m]
        for j in range(len(avgs)):
          avgs[j]/=len(bestmatches[i])
        clusters[i]=avgs
      
  return bestmatches

###Tanimoto系数，代表交集与并集的比率
def tanamoto(v1,v2):
  c1,c2,shr=0,0,0
  
  for i in range(len(v1)):
    if v1[i]!=0: c1+=1 # in v1
    if v2[i]!=0: c2+=1 # in v2
    if v1[i]!=0 and v2[i]!=0: shr+=1 # in both
  
  return 1.0-(float(shr)/(c1+c2-shr))

###多维缩放 为多维的数据集以二维形式表达
#先生成好节点，然后根据距离值不断调整 / 每个节点的移动都是其他所有节点施加的拖或者拉的综合效应·
#直到无法通过移动减小误差
def scaledown(data,distance=pearson,rate=0.01):
  #接收一个数据向量作为参数，输出只含两列的向量，即x，y值
  n=len(data) 
  # The real distances between every pair of items 每对数据项之间的真实距离
  realdist=[[distance(data[i],data[j]) for j in range(n)] 
             for i in range(n)]
             #一个二维矩阵，列行都是数据项，值代表两项间距离

  # Randomly initialize the starting points of the locations in 2D随机初始化起始位置
  loc=[[random.random(),random.random()] for i in range(n)] #生成一个随机的n项二维坐标
  fakedist=[[0.0 for j in range(n)] for i in range(n)]#二维化数据中两数据间距离
  
  lasterror=None
  for m in range(0,1000): #m值无意义，仅仅是代表最大训练1000步，因为函数中有判断break的条件
    # Find projected distances 寻找投影后的距离
    for i in range(n): #计算两个二维化后的数据项的距离
      for j in range(n):
        fakedist[i][j]=sqrt(sum([pow(loc[i][x]-loc[j][x],2) 
                                 for x in range(len(loc[i]))])) #这里为啥不直接用x,y而是len的形式？
  
    # Move points
    grad=[[0.0,0.0] for i in range(n)] #表示每一项二维化数据x,y轴上需要移动的距离
    
    totalerror=0 #总误差 判断是否停止
    for k in range(n): #遍历每个数据项，根据到其他点的位置误差来分别移动
      for j in range(n): #遍历k到其他项的距离
        if j==k: continue #相等时表示到自己的距离所以继续
        # The error is percent difference between the distances 误差值等于目标距离与当前距离差值百分比
        errorterm=(fakedist[j][k]-realdist[j][k])/realdist[j][k] #由真实距离判断二维化后的距离误差 除以真实距离 归一化
        
        # Each point needs to be moved away from or towards the other每个节点都按误差比例移动
        # point in proportion to how much error it has
        grad[k][0]+=((loc[k][0]-loc[j][0])/fakedist[j][k])*errorterm
        grad[k][1]+=((loc[k][1]-loc[j][1])/fakedist[j][k])*errorterm

        # Keep track of the total error记录总误差值
        totalerror+=abs(errorterm)
    print totalerror

    # If the answer got worse by moving the points, we are done如果节点移动后效果更坏则程序结束
    if lasterror and lasterror<totalerror: break
    lasterror=totalerror
    
    # Move each of the points by the learning rate times the gradient
    for k in range(n):
      loc[k][0]-=rate*grad[k][0] #乘以一个学习比率后再移动 / 为啥是-=？改成+=就不行了 不报错但是不生成图片 误差值也很大 
      loc[k][1]-=rate*grad[k][1]  #对了，因为是误差值，前面取过绝对值所以是要减去的

  return loc

#画图看效果
def draw2d(data,labels,jpeg='mds2d.jpg'):
  img=Image.new('RGB',(2000,2000),(255,255,255))
  draw=ImageDraw.Draw(img)
  for i in range(len(data)):
    x=(data[i][0]+0.5)*1000
    y=(data[i][1]+0.5)*1000
    draw.text((x,y),labels[i],(0,0,0))
  img.save(jpeg,'JPEG')  
