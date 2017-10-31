### 常用
* 读取自定义词典

```
jieba.load_userdict(r'./data/user_dict.txt') # file_name为自定义词典的路径  
#（格式：每行：词 [词频] [词性]） 中括号代表可选
```

### 实践
* [二元语法模型](./jieba_cut_ngram.py)
* [jieba分词不同模式](./jieba_cut.ipynb)
* [读取文本分词并存储](./read_save.ipynb)
* [词性标注](./cixing.py)
* [TF-IDF](./if-idf.py)
