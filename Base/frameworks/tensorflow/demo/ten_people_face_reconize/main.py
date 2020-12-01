#coding=utf-8
#http://www.jianshu.com/p/3e5ddc44aa56
#tensorflow 1.3.1
#python 3.6
import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import matplotlib.patches as patches
import numpy
from PIL import Image

#获取dataset
'''
数据材料
这是一个小型的人脸数据库，一共有40个人，每个人有10张照片作为样本数据。这些图片都是黑白照片，意味着这些图片都只有灰度0-255，没有rgb三通道。
于是我们需要对这张大图片切分成一个个的小脸。整张图片大小是1190 × 942，一共有20 × 20张照片。那么每张照片的大小就是
（1190 / 20）× （942 / 20）= 57 × 47 （大约，因为每张图片之间存在间距）。
'''
def load_data(dataset_path):
    img = Image.open(dataset_path)
    # 定义一个20 × 20的训练样本，一共有40个人，每个人都10张样本照片
    #print(img) #<PIL.GifImagePlugin.GifImageFile image mode=P size=942x1140 at 0x101AC2CF8>
    img_ndarray = np.asarray(img, dtype='float64') / 256 #转为矩阵 归一化
    # 记录脸数据矩阵，57 * 47为每张脸的像素矩阵
    faces = np.empty((400, 57 * 47)) #存储单个face
    for row in range(20):
        for column in range(20):
            faces[20 * row + column] = np.ndarray.flatten(
                img_ndarray[row * 57: (row + 1) * 57, column * 47 : (column + 1) * 47]
            )

    label = np.zeros((400, 40)) #onehot
    for i in range(40):
        label[i * 10: (i + 1) * 10, i] = 1
        #每个人有10张 所以间隔10 同一个人同一位 置1

    # 将数据分成训练集，验证集，测试集
    train_data = np.empty((320, 57 * 47))
    train_label = np.zeros((320, 40))
    vaild_data = np.empty((40, 57 * 47))
    vaild_label = np.zeros((40, 40))
    test_data = np.empty((40, 57 * 47))
    test_label = np.zeros((40, 40))

    for i in range(40):
        #每个人的10张照片 前8张训练 第9张验证 第十张测试
        train_data[i * 8: i * 8 + 8] = faces[i * 10: i * 10 + 8]
        train_label[i * 8: i * 8 + 8] = label[i * 10: i * 10 + 8]

        vaild_data[i] = faces[i * 10 + 8]
        vaild_label[i] = label[i * 10 + 8]

        test_data[i] = faces[i * 10 + 9]
        test_label[i] = label[i * 10 + 9]

    train_data = train_data.astype('float32')
    vaild_data = vaild_data.astype('float32')
    test_data = test_data.astype('float32')

    return [
        (train_data, train_label),
        (vaild_data, vaild_label),
        (test_data, test_label)
    ]

def convolutional_layer(data, kernel_size, bias_size, pooling_size):
    kernel = tf.get_variable("conv", kernel_size, initializer=tf.random_normal_initializer())
    bias = tf.get_variable('bias', bias_size, initializer=tf.constant_initializer(0))

    conv = tf.nn.conv2d(data, kernel, strides=[1, 1, 1, 1], padding='SAME')
    #一般而言，对于输入张量（input tensor）有四维信息：[batch, height, width, channels]
    # strides参数确定了滑动窗口在各个维度上移动的步数。一种常用的经典设置就是要求，strides[0]=strides[3]=1。
    # strides[0] = 1，也即在 batch 维度上的移动为 1，也就是不跳过任何一个样本，否则当初也不该把它们作为输入（input）
    # strides[3] = 1，也即在 channels 维度上的移动为 1，也就是不跳过任何一个颜色通道；
    linear_output = tf.nn.relu(tf.add(conv, bias))
    pooling = tf.nn.max_pool(linear_output, ksize=pooling_size, strides=pooling_size, padding="SAME")
    #pooling_size同上 [1,2,2,1]即每次移动两格
    return pooling

def linear_layer(data, weights_size, biases_size):
    weights = tf.get_variable("weigths", weights_size, initializer=tf.random_normal_initializer())
    biases = tf.get_variable("biases", biases_size, initializer=tf.constant_initializer(0))
    return tf.add(tf.matmul(data, weights), biases)

def convolutional_neural_network(data):
    # 根据类别个数定义最后输出层的神经元
    n_ouput_layer = 40

    kernel_shape1=[5, 5, 1, 32]
    #卷积核. 类型和input必须相同，4维tensor, [filter_height, filter_width, in_channels, out_channels],如[5,5,3,32]
    kernel_shape2=[5, 5, 32, 64]
    full_conn_w_shape = [15 * 12 * 64, 1024]
    out_w_shape = [1024, n_ouput_layer]

    bias_shape1=[32]
    bias_shape2=[64]
    full_conn_b_shape = [1024]
    out_b_shape = [n_ouput_layer]

    data = tf.reshape(data, [-1, 57, 47, 1])

    # 经过第一层卷积神经网络后，得到的张量shape为：[batch, 29, 24, 32]
    with tf.variable_scope("conv_layer1") as layer1:
        layer1_output = convolutional_layer(
            data=data,
            kernel_size=kernel_shape1,
            bias_size=bias_shape1,
            pooling_size=[1, 2, 2, 1]
        )
    # 经过第二层卷积神经网络后，得到的张量shape为：[batch, 15, 12, 64]
    with tf.variable_scope("conv_layer2") as layer2:
        layer2_output = convolutional_layer(
            data=layer1_output,
            kernel_size=kernel_shape2,
            bias_size=bias_shape2,
            pooling_size=[1, 2, 2, 1]
        )
    with tf.variable_scope("full_connection") as full_layer3:
        # 讲卷积层张量数据拉成2-D张量只有有一列的列向量
        layer2_output_flatten = tf.contrib.layers.flatten(layer2_output)
        layer3_output = tf.nn.relu(
            linear_layer(
                data=layer2_output_flatten,
                weights_size=full_conn_w_shape,
                biases_size=full_conn_b_shape
            )
        )
        # layer3_output = tf.nn.dropout(layer3_output, 0.8)
    with tf.variable_scope("output") as output_layer4:
        output = linear_layer(
            data=layer3_output,
            weights_size=out_w_shape,
            biases_size=out_b_shape
        )

    return output

def train_facedata(dataset, model_dir,model_path):
    batch_size = 40

    train_set_x = dataset[0][0]
    train_set_y = dataset[0][1]
    valid_set_x = dataset[1][0]
    valid_set_y = dataset[1][1]
    test_set_x = dataset[2][0]
    test_set_y = dataset[2][1]

    X = tf.placeholder(tf.float32, [batch_size, 57 * 47])
    Y = tf.placeholder(tf.float32, [batch_size, 40])

    predict = convolutional_neural_network(X)
    cost_func = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=predict, labels=Y))
    optimizer = tf.train.AdamOptimizer(1e-2).minimize(cost_func)

    # 用于保存训练的最佳模型
    saver = tf.train.Saver()
    #model_dir = './model'
    #model_path = model_dir + '/best.ckpt'
    with tf.Session() as session:
        # 若不存在模型数据，需要训练模型参数
        if not os.path.exists(model_path + ".index"):
            session.run(tf.global_variables_initializer())
            best_loss = float('Inf')
            for epoch in range(20):
                epoch_loss = 0
                for i in range((int)(np.shape(train_set_x)[0] / batch_size)):
                    x = train_set_x[i * batch_size: (i + 1) * batch_size]
                    y = train_set_y[i * batch_size: (i + 1) * batch_size]
                    _, cost = session.run([optimizer, cost_func], feed_dict={X: x, Y: y})
                    epoch_loss += cost

                print(epoch, ' : ', epoch_loss)
                if best_loss > epoch_loss:
                    best_loss = epoch_loss
                    if not os.path.exists(model_dir):
                        os.mkdir(model_dir)
                        print("create the directory: %s" % model_dir)
                    save_path = saver.save(session, model_path)
                    print("Model saved in file: %s" % save_path)

        # 恢复数据并校验和测试
        saver.restore(session, model_path)
        correct = tf.equal(tf.argmax(predict,1), tf.argmax(Y,1))
        #tf.argmax(predict,axis=1) 每行的argmax
        #tf.equal 返回同 shape的tensor，每一个值对应比较的两个元素该位置是否相等
        valid_accuracy = tf.reduce_mean(tf.cast(correct,'float'))
        #cast(x, dtype, name=None) 将x的数据格式转化成dtype
        #tf.reduce_mean(x)#如果不指定第二个参数，那么就在所有的元素中取平均值 #指定第二个参数为0，则第一维的元素取平均值，即每一列求平均值
        print('valid set accuracy: ', valid_accuracy.eval({X: valid_set_x, Y: valid_set_y}))
        '''
        tensorflow有两种方式：Session.run和 Tensor.eval，
        如果你有一个Tensor t，在使用t.eval()时，等价于：tf.get_default_session().run(t). 
        你可以使用sess.run()在同一步获取多个tensor中的值，使用Tensor.eval()时只能在同一步当中获取一个tensor值，
        并且每次使用 eval 和 run时，都会执行整个计算图。
        #这里上面这句等价于 print('valid set accuracy: ', session.run(valid_accuracy,feed_dict={X: valid_set_x, Y: valid_set_y}))
        '''
        test_pred = tf.argmax(predict, 1).eval({X: test_set_x})
        test_true = np.argmax(test_set_y, 1)
        test_correct = correct.eval({X: test_set_x, Y: test_set_y})
        incorrect_index = [i for i in range(np.shape(test_correct)[0]) if not test_correct[i]]
        for i in incorrect_index:
            print('picture person is %i, but mis-predicted as person %i'
                %(test_true[i], test_pred[i]))
        plot_errordata(incorrect_index, "olivettifaces.gif")


#画出在测试集中错误的数据
def plot_errordata(error_index, dataset_path):
    img = mpimg.imread(dataset_path)
    plt.imshow(img)
    currentAxis = plt.gca()
    for index in error_index:
        row = index // 2
        column = index % 2
        currentAxis.add_patch(
            patches.Rectangle(
                xy=(
                     47 * 9 if column == 0 else 47 * 19,
                     row * 57
                    ),
                width=47,
                height=57,
                linewidth=1,
                edgecolor='r',
                facecolor='none'
            )
    )
    plt.savefig("result.png")
    plt.show()


def main():
    dataset_path = "olivettifaces.gif"
    data = load_data(dataset_path)
    model_dir = './model'
    model_path = model_dir + '/best.ckpt'
    train_facedata(data, model_dir, model_path)

if __name__ == "__main__" :
    main()