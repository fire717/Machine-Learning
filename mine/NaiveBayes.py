# coding: utf-8

#贝叶斯定理：P(A|B) = P(B|A) P(A) / P(B)

#data
#判断一句话的情感 1:positive 0:negtive
#语料库 、 假如已经分好词，且为了简化，只取几个词
x = [['happy','feel','movie'],
     ['happy','sad','not'],
     ['new','out','dinner'],
     ['happy','sunny','play'],
     ['sad','cry','movie']]

y = [1,0,1,1,0]


# In[19]:


#train
#计算每个词的概率
num_p = 0
num_n = 0
pos={}
neg={}
for l in y:
    if l == 1:
        num_p+=1
    else:
        num_n+=1
for i in range(len(x)):
    if y[i] == 1:
        for j in range(len(x[0])):
            #dict.get(key, default=None) 返回给定键的值。如果键不可用，则返回默认值None。
            pos[x[i][j]] = pos.get(x[i][j], 0)+1/num_p
    else:
        print(i)
        for k in range(len(x[0])):
            neg[x[i][k]] = neg.get(x[i][k], 0)+1/num_n
            print(neg)


# In[20]:


#predict
sentence = 'I am not happy even the weather is sunny cause I saw a sad movie'
#提取关键词(对应词库的)
test = ['not','happy','sunny','sad','movie']

P1=1
P2=1
for i in test:
    if i in pos:
        P1 *= pos[i] 
    if i in neg:
        P2 *= neg[i]

if P1>P2:
    print('positive')
else:
    print('negtive')

#output:negtive






