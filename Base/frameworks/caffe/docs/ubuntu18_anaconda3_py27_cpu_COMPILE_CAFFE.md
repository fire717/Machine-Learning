
先随便搜一个教程照着装，比如[这个](https://blog.csdn.net/lukaslong/article/details/81390276)
然后会遇到如下问题：


#### 1. recipe for target '.build_release/src/caffe/layers/detection_output_layer.o' failed
```
先protoc --version查看版本
然后conda install protobuf=x.x.x 
```

#### 2.Makefile:621: recipe for target '.build_release/tools/convert_imageset.bin' failed
```
conda install py-opencv=3.4.2
```

#### 3.fatal error: caffe/proto/caffe.pb.h: 没有那个文件或目录
```
In the directory you installed Caffe to
protoc src/caffe/proto/caffe.proto --cpp_out=.
mkdir include/caffe/proto
mv src/caffe/proto/caffe.pb.h include/caffe/proto
```

#### 4.libprotobuf.so.19: cannot open shared object file: No such file or directory
```
sudo find / -name libprotobuf.so.19
发现确实存在libprotobuf.so.19（备注libprotobuf.so.19是一个软链接文件）
解决办法：
sudo cp xx/xx/libprotobuf.so.19.0.0 /usr/local/lib/
sudo ln -s /usr/local/lib/libprotobuf.so.19.0.0 /usr/local/lib/libprotobuf.so.19

export LD_LIBRARY_PATH=/usr/local/lib
```

#### 5.ImportError: libopencv_core.so.3.4: cannot open shared object file: No such file or directory
```
sudo find / -name "libopencv_core.so.3.4*"
Then got the result: /usr/local/lib/libopencv_core.so.3.2.
Create a file called /etc/ld.so.conf.d/opencv.conf 
 write to it the path to the folder where the binary is stored.
For example, I wrote /usr/local/lib/ to my opencv.conf file.
Run the command line as follows.
sudo ldconfig -v
```

#### 6.ImportError: 'No module named skimage.io'
```
pip install scikit-image
```

#### 7.TypeError: __new__() got an unexpected keyword argument 'serialized_options'
```
pip install -U protobuf
```


最后需要加入环境变量export PYTHONPATH=~/caffe-ssd/python:$PYTHONPATH
