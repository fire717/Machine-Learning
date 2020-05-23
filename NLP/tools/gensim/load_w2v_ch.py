#coding:utf-8

#py3 
#https://ai.tencent.com/ailab/nlp/embedding.html

from gensim.models.keyedvectors import KeyedVectors


file = "Tencent_AILab_ChineseEmbedding.txt"


# with open(file,"r",encoding="utf-8") as f:
#     print(f.readline()) # 8824330 200
#     # print(f.readline())
#     # print(f.readline())


wv_from_text = KeyedVectors.load_word2vec_format(file, binary=False)

wv_from_text.most_similar(u"足球")
"""
[('足球运动', 0.8081263303756714), ('足球文化', 0.7661516070365906), ('足球发展', 0.7645934820175171), ('职业足球', 0.7609031200408936), ('足球教育', 0.7551054954528809), ('热爱足球', 0.7491205930709839), ('足球技术', 0.7459214925765991), ('踢球', 0.7441200017929077), ('世界足球', 0.7434529066085815), ('足球项目', 0.7409517765045166)]

"""
