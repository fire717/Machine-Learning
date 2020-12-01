#coding:utf-8
import clusters
blognames,words,data = clusters.readfile('blogdata.txt')
clust = clusters.hcluster(data)

#显示聚类树
#clusters.printclust(clust,labels=blognames)

#生成聚类图
#clusters.drawdendrogram(clust,blognames,jpeg='blogclust.jpg')

#多维缩放
coords = clusters.scaledown(data)
clusters.draw2d(coords,blognames,jpeg='blogs2d.jpg')
