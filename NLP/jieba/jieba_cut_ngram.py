#coding:utf-8
import jieba

ori_data = u'刘超是一个喜欢学习的好学生，你看，他正在学习呢。除了学习刘超还喜欢打游戏。'
print ori_data

def reform(sentence):
    #如果是以“。”结束的则将“。”删掉
    if sentence[-1] == u"。":
        sentence=sentence[:-1]
    #添加起始符BOS和终止符EOS 
    tmp = [u'、' ,u'，',u'：',u'。']
    for i in xrange(len(sentence)):
        if sentence[i] in tmp:
            sentence=sentence[:i]+'EEEBBB'+sentence[i+1:]
    sentence2="BBB"+sentence+"EEE"
    return sentence2

data1 = reform(ori_data)
print data1

#分词并统计词频
def segmentation(sentence,lists=[],dicts={}):
    jieba.suggest_freq(u"BBB", True)
    jieba.suggest_freq(u"EEE", True)
    jieba.suggest_freq(u"刘超", True)
    #分词
    sentence = jieba.cut(sentence,cut_all=False,HMM=False)
    #组合
    format_sentence=",".join(sentence)
    lists=format_sentence.split(",")     
    for word in lists:
        if word not in dicts:
            dicts[word]=1
        else:
            dicts[word]+=1 
    return lists


dict1 = {}
t = segmentation(data1,lists=[],dicts=dict1)
for x in t:
    print x.encode('utf-8')
#输出词频，同时去除一些杂词  
badwords = []
#badwords = [u'的',u'是',u'呢',u'还',u'BBB',u'EEE']
for key in dict1.keys():
    if key not in badwords:
        print key.encode('utf-8'),':',dict1[key]
        
        
test1 = u'刘超喜欢学习'
test2 = u'学习喜欢刘超'

def segmentation(sentence,lists=[]):
    jieba.suggest_freq(u"刘超", True)
    #分词
    sentence = jieba.cut(sentence,cut_all=False,HMM=False)
    #组合
    format_sentence=",".join(sentence)
    lists=format_sentence.split(",")     
    return lists

test1 = segmentation(test1)
test2 = segmentation(test2)

#比较两个数列，二元语法
def compareList(ori_list,test_list):
    #申请空间
    count_list=[0]*(len(test_list))
    #遍历测试的字符串
    for i in range(0,len(test_list)-1):
        #遍历语料字符串，且因为是二元语法，不用比较语料字符串的最后一个字符
        for j in range(0,len(ori_list)-2):                
            #如果测试的第一个词和语料的第一个词相等则比较第二个词
            if test_list[i]==ori_list[j]:
                if test_list[i+1]==ori_list[j+1]:
                    count_list[i]+=1
    return count_list

print compareList(t,test1)
print compareList(t,test2)

#计算概率    
def probability(test_list,count_list,ori_dict):
    flag=0
    #概率值为p
    p=1
    for key in test_list: 
        #数据平滑处理：加1法
        p*=(float(count_list[flag]+1)/float(ori_dict[key]+1))
        flag+=1
    return p
print probability(test1,compareList(t,test1),dict1)
print probability(test2,compareList(t,test2),dict1)
