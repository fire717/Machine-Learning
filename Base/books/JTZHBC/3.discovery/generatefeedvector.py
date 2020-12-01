#coding:utf-8
import feedparser
import re

# 返回一个RSS订阅源的标题和包含单词计数情况的字典
def getwordcounts(url):
  # Parse the feed 解析订阅源
  d=feedparser.parse(url)
  wc={}

  # Loop over all the entries循环遍历所有文章条目
  for e in d.entries:
    if 'summary' in e: summary=e.summary
    else: summary=e.description

    # Extract a list of words提取一个单词列表
    words=getwords(e.title+' '+summary)
    for word in words:
      wc.setdefault(word,0)
      #setdefault() 函数和get() 方法类似, 如果键不存在于字典中，将会添加键并将值设为默认值。
      wc[word]+=1
  return d.feed.title,wc

def getwords(html):
  # Remove all the HTML tags 去除所有HTML标记
  txt=re.compile(r'<[^>]+>').sub('',html)
  #re.sub(a,b,x)用作把x中的a替换为b，这里没有b，应该就是删除空格
  
  # Split words by all non-alpha characters利用所有非字母字符拆分出单词
  words=re.compile(r'[^A-Z^a-z]+').split(txt)

  # Convert to lowercase转化小写
  return [word.lower() for word in words if word!='']


apcount={}
wordcounts={}
feedlist=[line for line in file('feedlist.txt')] #循环遍历订阅源
#不用read直接读取txt文件
for feedurl in feedlist:
  try:
    title,wc=getwordcounts(feedurl) #得到每篇文章的词和次数
    wordcounts[title]=wc
    for word,count in wc.items(): #items()方法返回字典的(键，值)元组对的列表 / 试了下，不加items会报错
      apcount.setdefault(word,0)
      if count>1:
        apcount[word]+=1
  except:
    print 'Failed to parse feed %s' % feedurl

wordlist=[]
for w,bc in apcount.items(): #items()方法返回字典的(键，值)元组对的列表
  frac=float(bc)/len(feedlist)
  if frac>0.1 and frac<0.5: #去掉出现频率太高太低的词
    wordlist.append(w)

out=file('blogdata1.txt','w')
out.write('Blog')
for word in wordlist: out.write('\t%s' % word)
out.write('\n')
for blog,wc in wordcounts.items():
  print blog
  out.write(blog)
  for word in wordlist:
    if word in wc: out.write('\t%d' % wc[word])
    else: out.write('\t0')
  out.write('\n')
