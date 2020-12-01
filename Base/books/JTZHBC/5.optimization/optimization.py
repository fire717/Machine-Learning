#coding:utf-8
#使用随机优化的技术解决协作类问题
#典型应用场景是，存在大量可能的题解以至于我们无法一一尝试的情况。
import time
import random
import math

people = [('Seymour','BOS'),
          ('Franny','DAL'),
          ('Zooey','CAK'),
          ('Walt','MIA'),
          ('Buddy','ORD'),
          ('Les','OMA')]
# Laguardia
destination='LGA'

flights={}
# 载入数据到字典中
for line in file('schedule.txt'):
  origin,dest,depart,arrive,price=line.strip().split(',') #分隔赋值
  flights.setdefault((origin,dest),[])

  # Add details to the list of possible flights 把航班详情添加到列表中
  flights[(origin,dest)].append((depart,arrive,int(price)))

def getminutes(t): #计算时间段类分钟数
  x=time.strptime(t,'%H:%M') #strptime() 函数根据指定的格式把一个时间字符串解析为时间元组
  return x[3]*60+x[4]

def printschedule(r): #把决定搭乘的航班(一个数字列表)打印成表格
  for d in range(len(r)/2):
    name=people[d][0]
    origin=people[d][1]
    out=flights[(origin,destination)][int(r[d])]
    ret=flights[(destination,origin)][int(r[d+1])]
    print '%10s%10s %5s-%5s $%3s %5s-%5s $%3s' % (name,origin,
                                                  out[0],out[1],out[2],
                                                  ret[0],ret[1],ret[2])

def schedulecost(sol): #成本函数 计算不同航班计划的成本
  totalprice=0
  latestarrival=0
  earliestdep=24*60

  for d in range(len(sol)/2):
    # Get the inbound and outbound flights 得到往程航班和返程航班
    origin=people[d][1]
    outbound=flights[(origin,destination)][int(sol[d])]
    returnf=flights[(destination,origin)][int(sol[d+1])]
    
    # Total price is the price of all outbound and return flights 总价格等于所有往程航班和返程航班价格之和
    totalprice+=outbound[2]
    totalprice+=returnf[2]
    
    # Track the latest arrival and earliest departure 记录最晚到达时间和最早离开时间
    if latestarrival<getminutes(outbound[1]): latestarrival=getminutes(outbound[1])
    if earliestdep>getminutes(returnf[0]): earliestdep=getminutes(returnf[0])
  
  # Every person must wait at the airport until the latest person arrives.
  # They also must arrive at the same time and wait for their flights.
  totalwait=0  
  for d in range(len(sol)/2):
    origin=people[d][1]
    outbound=flights[(origin,destination)][int(sol[d])]
    returnf=flights[(destination,origin)][int(sol[d+1])]
    totalwait+=latestarrival-getminutes(outbound[1])
    totalwait+=getminutes(returnf[0])-earliestdep  

  # Does this solution require an extra day of car rental? That'll be $50!这个题解要求多付一天的汽车租用费用吗？如果是，则费用为50刀
  if latestarrival>earliestdep: totalprice+=50
  
  return totalprice+totalwait

###随机搜索
def randomoptimize(domain,costf):
    #domain是一个二元组列表，指定每个变量的最小最大值，这里是(0,9)
    #costf 成本函数
  best=999999999
  bestr=None
  for i in range(0,1000): #随机产生1000次猜测
    # Create a random solution
    r=[float(random.randint(domain[i][0],domain[i][1])) 
       for i in range(len(domain))]
    
    # Get the cost
    cost=costf(r)
    
    # Compare it to the best one so far
    if cost<best:
      best=cost
      bestr=r 
  return r

#爬山法
'''
类似于梯度下降，容易陷入局部最优解。解决方法一般是随机梯度下降：以多个随机生成的初始解开始运行数次，借此希望其中有一个解能逼近全局最优解
还有其他的避免局部最优解的方法：模拟退火算法，遗传算法。
'''
def hillclimb(domain,costf):
  # Create a random solution
  sol=[random.randint(domain[i][0],domain[i][1])
      for i in range(len(domain))]
  # Main loop
  while 1:
    # Create list of neighboring solutions 创建相邻解的列表
    neighbors=[]
    
    for j in range(len(domain)):
      # One away in each direction 在每个方向上相对于原值偏离一点
      if sol[j]>domain[j][0]:
        neighbors.append(sol[0:j]+[sol[j]+1]+sol[j+1:])
      if sol[j]<domain[j][1]:
        neighbors.append(sol[0:j]+[sol[j]-1]+sol[j+1:])

    # See what the best solution amongst the neighbors is
    current=costf(sol)
    best=current
    for j in range(len(neighbors)):
      cost=costf(neighbors[j])
      if cost<best:
        best=cost
        sol=neighbors[j]

    # If there's no improvement, then we've reached the top
    if best==current:
      break
  return sol

#模拟退火算法
#差别：新的值成本更高依旧可能成为当前题解 、且由公式决定是否接受
def annealingoptimize(domain,costf,T=10000.0,cool=0.95,step=1):
    #初始温度 / 冷却率 / 推进值
  # Initialize the values randomly
  vec=[float(random.randint(domain[i][0],domain[i][1])) 
       for i in range(len(domain))]
  
  while T>0.1:
    # Choose one of the indices选择一个索引值
    i=random.randint(0,len(domain)-1)

    # Choose a direction to change it
    dir=random.randint(-step,step)

    # Create a new list with one of the values changed创建一个代表题解的新列表，改变其中一个值
    vecb=vec[:]
    vecb[i]+=dir
    if vecb[i]<domain[i][0]: vecb[i]=domain[i][0]
    elif vecb[i]>domain[i][1]: vecb[i]=domain[i][1]

    # Calculate the current cost and the new cost
    ea=costf(vec)
    eb=costf(vecb)

    p=pow(math.e,(-eb-ea)/T) ###更高成本的解被接受的概率的公式

    # Is it better, or does it make the probability 它是更好地解吗？或者是趋向最优解的可能的临界解吗？
    # cutoff?
    if (eb<ea or random.random()<p):
      vec=vecb      

    # Decrease the temperature 降低温度
    T=T*cool
  return vec


###遗传算法
'''
先随机生成一组解，称之为种群。在优化过程每一步，算法计算整个种群的成本函数，从而得到一个有关题解的有序列表。
然后进行排序，把最顶端的题解加入新种群，这一过程称为精英选拔。
新种群的余下部分是由修改最优解后形成的全新解。
两种修改题解的办法：
1.变异 ：对一个既有解进行微小的简单的随机的改变。
2.交叉 / 配对：选取最优解中的两个解，然后按某种方式进行结合。

形成相同数目的新的种群，然后不断迭代。
'''
def geneticoptimize(domain,costf,popsize=50,step=1,
                    mutprob=0.2,elite=0.2,maxiter=100):
    #可选参数：popsize种群大小 
    #           mutprob种群的新成员是由变异而非交叉得来的概率
    #           elite种群中被认为是最优解且被允许传入下一代的部分
    #           需要运行多少代
  # Mutation Operation 变异操作
  def mutate(vec):
    i=random.randint(0,len(domain)-1)
    if random.random()<0.5 and vec[i]>domain[i][0]:
      return vec[0:i]+[vec[i]-step]+vec[i+1:] 
    elif vec[i]<domain[i][1]:
      return vec[0:i]+[vec[i]+step]+vec[i+1:]
  
  # Crossover Operation交叉操作
  def crossover(r1,r2):
    i=random.randint(1,len(domain)-2)
    return r1[0:i]+r2[i:]

  # Build the initial population构造初始种群
  pop=[]
  for i in range(popsize):
    vec=[random.randint(domain[i][0],domain[i][1]) 
         for i in range(len(domain))]
    pop.append(vec)
  
  # How many winners from each generation?每一代有多少胜出者
  topelite=int(elite*popsize)
  
  # Main loop 
  for i in range(maxiter):
    scores=[(costf(v),v) for v in pop]
    scores.sort()
    ranked=[v for (s,v) in scores]
    
    # Start with the pure winners从纯粹的胜出者开始
    pop=ranked[0:topelite]
    
    # Add mutated and bred forms of the winners添加变异和配对后的胜出者
    while len(pop)<popsize:
      if random.random()<mutprob:

        # Mutation变异
        c=random.randint(0,topelite)
        pop.append(mutate(ranked[c]))
      else:
      
        # Crossover交叉
        c1=random.randint(0,topelite)
        c2=random.randint(0,topelite)
        pop.append(crossover(ranked[c1],ranked[c2]))
    
    # Print current best score
    print scores[0][0]
    
  return scores[0][1]
