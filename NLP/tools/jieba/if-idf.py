# ###基于 TF-IDF 算法的关键词提取

# import jieba.analyse
# jieba.analyse.extract_tags(sentence, topK = 20, withWeight = False, allowPOS = ())
# sentence:待提取的文本。
# topK:返回几个 TF/IDF 权重最大的关键词，默认值为20。
# withWeight:是否一并返回关键词权重值，默认值为False。
# allowPOS:仅包括指定词性的词，默认值为空，即不进行筛选。
# jieba.analyse.TFIDF(idf_path=None) 新建 TFIDF 实例，idf_path 为 IDF 频率文件。

# optparse模块OptionParser学习
# optparse是专门在命令行添加选项的一个模块。

# In[2]:

from optparse import OptionParser
MSG_USAGE = "myprog[ -f ][-s ] arg1[,arg2..]"
optParser = OptionParser(MSG_USAGE)
#以上，产生一个OptionParser的物件optParser。传入的值MSG_USAGE可被调用打印命令时显示出来。

optParser.add_option("-f","--file",action = "store",type="string",dest = "fileName")
optParser.add_option("-v","--vison", action="store_false", dest="verbose",default='gggggg',
                     help="make lots of noise [default]")
#调用OptionParser.add_option()添加选项，add_option()参数说明：
#action:存储方式，分为三种store, store_false, store_true
#type:类型
#dest:存储的变量
#default:默认值
#help:帮助信息

fakeArgs = ['-f','file.txt','-v','good luck to you', 'arg2', 'arge']
options, args = optParser.parse_args(fakeArgs)
print(options.fileName)
print(options.verbose)
print(options)
print(args)
#调用OptionParser.parse_args()剖析并返回一个directory和一个list
#parse_args()说明:
#如果没有传入参数，parse_args会默认将sys.argv[1:]的值作为默认参数。这里我们将fakeArgs模拟输入的值。
#从返回结果中可以看到，
#options为是一个directory,它的内容fakeArgs为“参数/值 ”的键值对。
#args 是一个list，它的内容是fakeargs除去options后，剩余的输入内容。
#options.version和options.fileName都取到与options中的directory的值。

print(optParser.print_help())
#输出帮助信息
#optParser.print_help()说明：
#1、最开始的的MSG_USAGE的值:在这个地方显示出来了。
#2、自动添加了-h这个参数。


# In[14]:
################################
#######      main     ##########
###############################
import jieba.analyse as anl
f = open("C:\\Users\\Fire\\Desktop\\p.txt", "r").read()
seg = anl.extract_tags(f, topK = 20, withWeight = True)
for tag, weight in seg:
    print("%s %s" %(tag, weight))


# 关键词提取所使用逆向文件频率（IDF）文本语料库可以切换成自定义语料库的路径。
# jieba.analyse.set_idf_path(file_name) #file_name为自定义语料库的路径
# 如：jieba.analyse.set_idf_path("../extra_dict/idf.txt.big")
# .big文件一般是游戏中的文件，比较常见的用途是装载游戏的音乐、声音等文件。
# 测试可以使用txt，每一行为一段话
# 
# 关键词提取所使用停用词（Stop Words）文本语料库可以切换成自定义语料库的路径。
# jieba.analyse.set_stop_words(file_name) #file_name为自定义语料库的路径。
# 如：jieba.analyse.set_stop_words("../extra_dict/stop_words.txt")

'''
#output:
file.txt
False
{'verbose': False, 'fileName': 'file.txt'}
['good luck to you', 'arg2', 'arge']
Usage: myprog[ -f ][-s ] arg1[,arg2..]

Options:
  -h, --help            show this help message and exit
  -f FILENAME, --file=FILENAME
  -v, --vison           make lots of noise [default]
None
电影院 0.4067977462134147
被困 0.35729738628721947
山体 0.1899807393285366
变形 0.1873677446897561
15 0.17494781711560975
碧罗 0.17494781711560975
郭健摄 0.17494781711560975
贡山 0.17127004499414636
垮塌 0.1637945480619512
救援 0.15384430228839024
民众 0.13798957229912195
滑坡 0.13725203524287805
图为 0.1360318165754634
数字 0.1263477027307317
12 0.11663187807707316
11 0.11663187807707316
消防大队 0.11315212252682927
消防官兵 0.10797526154048781
三组 0.10421270075219512
现场 0.092375911824
'''
