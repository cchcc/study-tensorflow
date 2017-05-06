# encoding=utf8
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
import tensorflow as tf

# 앞의 방법을 행렬을 이용한 방법으로 변경

x_data = [ [73,80,75]
           ,[93,88,93]
           ,[89,91,90]
           ,[96,98,100]
           ,[73,66,70]]
y_data = [[152],[185],[180],[196],[142]]

X = tf.placeholder(tf.float32, shape=[None, 3])  # 여기서 None 의 의미는 행이 몇개든 늘어날수 있다는걸 의미
Y = tf.placeholder(tf.float32, shape=[None, 1])

W = tf.Variable(tf.random_normal([3, 1]))
b = tf.Variable(tf.random_normal([1]))

hypothesis = tf.matmul(X, W) + b  # 행렬의 곱

cost = tf.reduce_mean(tf.square(hypothesis - Y))
optimizer = tf.train.GradientDescentOptimizer(learning_rate=1e-5)
train = optimizer.minimize(cost)

s = tf.Session()
s.run(tf.global_variables_initializer())

for step in range(2001):
    # print(s.run([cost, hypothesis, train], feed_dict={x1:x1_data, x2:x2_data, x3:x3_data, Y:y_data}))
    s.run(train, feed_dict={X:x_data, Y:y_data})
    print(s.run([W, b]))


print(s.run(hypothesis, feed_dict={X:[[73,80,75]]}))
print(s.run(hypothesis, feed_dict={X:[[93,88,93]]}))