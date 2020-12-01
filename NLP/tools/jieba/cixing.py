import jieba.posseg as pseg
words = pseg.cut("迅速落实整改，报道称 河南省2017年护士执业资格考试已于2017年5月8日结束 模块消防站全景器材室多功能室图书室厨房")
for w in words:
    print("%s %s" %(w.word, w.flag))
    
    
'''
#output:
迅速 ad
落实 a
整改 v
， x
报道 v
称 v
...
'''
