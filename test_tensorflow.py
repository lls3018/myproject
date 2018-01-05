# -*- coding:utf-8 -*-
__author__ = 'leon'
import tensorflow as tf
import tensorflow.examples.tutorials.mnist.input_data


mnist = input_data.read_data_sets("MNIST_data", one_hot=True)
sess = tf.InteractiveSession()
sess.run(tf.initialize_all_variables())

# x为样本
x = tf.placeholder(tf.float32, [None, 784])
# y_为label，目标类别
y_ = tf.placeholder('float', [None, 10])
# w为模型定义权重
w = tf.Variable(tf.zeros([784, 10]))
# b为模型偏执
b = tf.Variable(tf.zeros([10]))
# y为模型输出，预测类别
y = tf.nn.softmax(tf.matual(x,w)+b)
# 定义损失函数，损失函数是目标类别与预测类别的交叉熵
cross_entropy = -tf.reduce_sum(y_*tf.log(y))
# 模型训练方法，使用最速下降法让交叉熵下降，步长为0.01
train_step = tf.train.GradientDescentOptimizer(0.01).minimize(cross_entropy)

for i in range(1000):
    # 每一次加载50个样本
    batch = mnist.train.next_batch(50)
    # 通过feed_dict, 将x和_y张量张量占位符用训练数据代替
    train_step.run(feed_dict={x: batch[0], y_: batch[1]})

# 判断预测准确性
correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))
# 计算准确率
accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
print accuracy.eval(feed_dict={x: mnist.test.images, y_: mnist.test.labels})


def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)


def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)


def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1,1,1,1], padding='SAME')


def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')





#一.准备数据
    #1.1下载训练数据
    #1.2输入占位符,后续实际的训练数据需要输入到这里
#二.构建图表
    #2.1.推理
    #2.2.损失函数
    #2.3.训练
#三.训练模型
    #3.1图表
    #3.2会话
    #3.3训练循环
#四.模型评估
    #4.1构建评估图表
    #4.2评估图表的输出
    



































