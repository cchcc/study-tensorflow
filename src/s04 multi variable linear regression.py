# encoding=utf8

# The TensorFlow library wasn't compiled to use SSE4.1 instructions... 워닝 끔
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

import tensorflow as tf


# 인풋이 여러개일때 -> 공식이 매우 길어짐 -> 이걸 좀더 간결하게 표현하자면
# 행렬(matrix)의 곱셈(첫항의 가로열 * 둘째항의 세로행)을 이용한 표현식 H(X) = XW  대문자, X가 앞에옴(구현에서도 이렇게 사용함)
# b를 행렬에 포함 시킴 둘째는 1
# 하나의 케이스를 instance 라 함
# XW 에서 W 의 크기를 결정하는 방법 : [5 , 3] -> 5개의 인스턴스, 3개의 x(인풋)   [3 , 1]  =  [5 , 1]  (1은 결과의 갯수) 5 by 3
# W크기 : [입력갯수, 출렷갯수]
# [n , 3] (가변적인 크기인 n은 -1 또는 tensorflow에서는 none 으로 표현)
# 인풋이 여러개이면 행렬의 곱을 이용함!
#
# 이론을 표현 할때 : H(x) = Wx + b
# 실제 구현에서는 : H(x) = XW

x1_data = [73,93,89,96,73]
x2_data = [80,88,91,98,66]
x3_data = [75,93,90,100,70]
y_data = [152,185,180,196,142]

x1 = tf.placeholder(tf.float32)
x2 = tf.placeholder(tf.float32)
x3 = tf.placeholder(tf.float32)
Y = tf.placeholder(tf.float32)

w1 = tf.Variable(tf.random_normal([1]))
w2 = tf.Variable(tf.random_normal([1]))
w3 = tf.Variable(tf.random_normal([1]))
b = tf.Variable(tf.random_normal([1]))

hypothesis = x1*w1 + x2*w2 + x3*w3 + b

# 위에 방법은 인풋이 늘어나면 코드가 계속 늘어나므로 좋은 방법이 아님 -> 행렬을 이용해보자

cost = tf.reduce_mean(tf.square(hypothesis - Y))
optimizer = tf.train.GradientDescentOptimizer(learning_rate=1e-5)
train = optimizer.minimize(cost)

s = tf.Session()
s.run(tf.global_variables_initializer())

for step in range(2001):
    # print(s.run([cost, hypothesis, train], feed_dict={x1:x1_data, x2:x2_data, x3:x3_data, Y:y_data}))
    s.run(train, feed_dict={x1:x1_data, x2:x2_data, x3:x3_data, Y:y_data})
    print(s.run([w1, w2, w3, b]))


print(s.run(hypothesis, feed_dict={x1:73, x2:80, x3:75}))
print(s.run(hypothesis, feed_dict={x1:93, x2:88, x3:93}))