
# coding: utf-8

# In[1]:



#==================== Part 0: Basic Function ====================
with open('D:\ex2data1.txt', 'r') as f:  
    data = f.readlines()  #txt中所有字符串读入data  

datamat = []
for line in data:  
    odom = line.strip().split(',')        #将单个数据分隔开存好  
    numbers_float = map(float, odom) #转化为浮点数  
    datamat.append(numbers_float)

#print datamat
import numpy as np

datanp = np.array(datamat)
#print datanp
x = datanp[:,(0,1)].reshape((100,2))  
y = datanp[:,2].reshape((100,1))
lenY = len(y)
X = np.hstack((np.ones((lenY,1)),x)) 


# In[2]:


#%% ==================== Part 1: Plotting ====================
#We start the exercise by first plotting the data to understand the the problem we are working with.
import matplotlib.pyplot as plt

def plotData(x,y):
    fig = plt.figure()  
    ax= fig.add_subplot(111) #使画在一个图上
    
    pos = np.where(y[:,0]==0) #y为类似矩阵形式，所以要再取第一列
    neg = np.where(y[:,0]==1)
    ax1 = plt.scatter(x[pos,0], x[pos,1], marker = 'x', color = 'm')  
    ax2 = plt.scatter(x[neg,0], x[neg,1], marker = 'o', color = 'r')  
    plt.xlabel('exam1 score') 
    plt.ylabel('exam2 score')
    
    plt.legend([ax1, ax2], ['Admitted', 'Not admitted'])
    plt.show()

plotData(x,y)


# In[3]:


from sklearn import datasets
from sklearn.cross_validation import train_test_split

from sklearn.linear_model import LogisticRegression
lr = LogisticRegression(C=1e9)
lr.fit(x, y)   #这里要用的x是原始的！不带1(x0)的！
print lr.coef_,lr.intercept_
final_theta = np.zeros((3,1))
final_theta[0] = lr.intercept_
final_theta[1],final_theta[2]= lr.coef_[0]
print final_theta


# In[4]:


#%% ============= Part 4: Visualizing J(theta_0, theta_1) =============
fig = plt.figure()  
ax= fig.add_subplot(111) #使画在一个图上
    
pos = np.where(y[:,0]==1) #y为类似矩阵形式，所以要再取第一列
neg = np.where(y[:,0]==0)
    
ax1 = plt.scatter(x[pos,0], x[pos,1], marker = 'x', color = 'm')  
ax2 = plt.scatter(x[neg,0], x[neg,1], marker = 'o', color = 'r')  
    
plt.xlabel('exam1 score') 
plt.ylabel('exam2 score')
    
plt.legend([ax1, ax2], ['Admitted', 'Not admitted'])


#plotX = [30,100] #范围a[2]
plotX = np.arange(30,100,1)
plotY = (-final_theta[0]-final_theta[1]*plotX)/final_theta[2] #由0=w0x0+w1x1+w2x2推导，这里的y就是x2，x0=1 
#注意等于0！！！因为这是分类问题
plt.plot(plotX,plotY)#调用plot函数绘制得到由点生成的线条

    
plt.show()


# In[ ]:




