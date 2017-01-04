
import numpy as np
import cv2
import tensorflow as tf 


from tensorflow.examples.tutorials.mnist import input_data

def inverse_color(image):

    height,width = image.shape
    img2 = image.copy()

    for i in range(height):
        for j in range(width):
            img2[i,j] = (255-image[i,j]) 
    return img2

mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

x = tf.placeholder(tf.float32, [None, 784])
W = tf.Variable(tf.zeros([784,10]))
b = tf.Variable(tf.zeros([10]))

y = tf.nn.softmax(tf.matmul(x,W) + b)

y_ = tf.placeholder("float", [None,10])

cross_entropy = -tf.reduce_sum(y_*tf.log(y))

train_step = tf.train.GradientDescentOptimizer(0.01).minimize(cross_entropy)

init = tf.initialize_all_variables()

sess = tf.Session()
sess.run(init)

for i in range(1000):
  batch_xs, batch_ys = mnist.train.next_batch(100)
  sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})

correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))

accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))

print sess.run(accuracy, feed_dict={x: mnist.test.images, y_: mnist.test.labels})


#read any size pic
z = cv2.imread("2.png",0)
z = cv2.resize(z,(28,28),interpolation = cv2.INTER_CUBIC)
z=inverse_color(z)

image = np.reshape(z,[1,784],order='C')
#cant use tf.reshape() cause its output is a tensor while cant be feed 

x2 = tf.placeholder(tf.float32, [1, 784])
y2 = tf.nn.softmax(tf.matmul(x2,W) + b)
ans = tf.argmax(y2,1)
print sess.run(ans,feed_dict={x2:image,})
