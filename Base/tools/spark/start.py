from pyspark import SparkConf, SparkContext
conf = SparkConf().setMaster("local[*]").setAppName("First_App")
sc = SparkContext(conf=conf)

data = sc.parallelize(range(10))
ans = data.reduce(lambda x, y: x + y)
print (ans)

'''
output:

Setting default log level to "WARN".
To adjust logging level use sc.setLogLevel(newLevel). For SparkR, use setLogLevel(newLevel).
2018-05-16 17:08:22 WARN  Utils:66 - Service 'SparkUI' could not bind on port 4040. Attempting port 4041.

[Stage 0:>                                                          (0 + 4) / 4]
[Stage 0:==============>                                            (1 + 3) / 4]
[Stage 0:=============================>                             (2 + 2) / 4]
[Stage 0:============================================>              (3 + 1) / 4]
                                                                                
45
[Decode error - output not utf-8]
[Decode error - output not utf-8]
[Decode error - output not utf-8]
'''
