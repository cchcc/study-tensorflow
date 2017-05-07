# encoding=utf8

import tensorflow as tf
import numpy as np

xy = np.loadtxt('s06-2_zoo.csv',delimiter=',',dtype=np.float32)

x_data = xy[:, 0:-1]
y_data = xy[:, [-1]]

nb_classes = 7

Y = tf.placeholder(tf.int32, [None, 1])
# 결과값이 3일때 [0,0,0,1,0,0,0] 이런식으로 바꿔줌, 근데 결과가 배열을 하나 더 감싸서 리턴함(rank n +1)
Y_one_hot = tf.one_hot(Y, nb_classes)
# 그래서 배열을 한단계 벗겨줌  https://www.tensorflow.org/api_docs/python/tf/reshape
Y_one_hot_reshaped = tf.reshape(Y_one_hot, [-1, nb_classes])  # 원하는 shape 으로 변경함


X = tf.placeholder(tf.float32, [None, 16])

W = tf.Variable(tf.random_normal([16, nb_classes]))
b = tf.Variable(tf.random_normal([nb_classes]))

logits = tf.matmul(X, W) + b
hypothesis = tf.nn.softmax(logits)

# -tf.reduce_mean(Y * tf.log(hypothesis), axis=1)  이거와 아래함수는 같은거임
cost_softmax_cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=Y_one_hot_reshaped)
cost = tf.reduce_mean(cost_softmax_cross_entropy)

optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.1).minimize(cost)

# 예측결과와 실제결과를 비교해서 정확도를 계산해봄
prediction = tf.arg_max(hypothesis, 1)
prediction_result = tf.equal(prediction, tf.arg_max(Y_one_hot_reshaped, 1))
accuracy = tf.reduce_mean(tf.cast(prediction_result, tf.float32))

with tf.Session() as s:
    s.run(tf.global_variables_initializer())

    for step in range(2001):
        s.run(optimizer, feed_dict={X: x_data, Y: y_data})
        print(step, s.run([cost, accuracy], feed_dict={X: x_data, Y: y_data}))
