# encoding=utf8

# (Binary) classification 1 or 0?
# 둘중 하나의 경우로 분류를 하고 싶다 -> linear regression 을 이용할수 있을것 같다 -> 학습과정에서 트레인 데이터 값이 극단으로 차이가 나면?
# -> 그래프 기울기가 틀려져서 기준이 틀려져 버린다 -> 그러면 어떤 인풋에 대해 무조건 0 or 1 로 만들어 주는 함수가 없을까?
# -> 그래서 나온게 sigmoid(logistic) function : g(z)

# logistic hypothesis

# cost function 의 함수 모양이 리니어랑 틀림 local minimum, global minimum 이 있고
# local minimum 에서 기울기가 0이 나오는 지점이 있음
# -> x와 y차의 제곱으로 만든 cost function 으로는 gradient descent 를 쓸수 없다!

# 로그함수를 이용함, y 가 0 일때와 1 일때의 경우 공식이 틀리다 -> 이걸 한줄로 합침

import tensorflow as tf

x_data = [[1,2],[2,3],[3,1],[4,3],[5,3],[6,2]]
y_data = [[0],[0],[0],[1],[1],[1]]

X = tf.placeholder(tf.float32, shape=[None, 2])
Y = tf.placeholder(tf.float32, shape=[None, 1])

W = tf.Variable(tf.random_normal([2,1]))  # 2 는 인풋의 갯수, 1은 아웃풋의 갯수
b = tf.Variable(tf.random_normal([1]))

hypothesis = tf.sigmoid(tf.matmul(X, W) + b)  # tf.div(1., 1 + tf.exp(tf.matmul(X, W) + b))

cost = -tf.reduce_mean(Y*tf.log(hypothesis) + (1-Y)*tf.log(1-hypothesis))

train = tf.train.GradientDescentOptimizer(learning_rate=0.01).minimize(cost)

predicted = tf.cast(hypothesis > 0.5, dtype=tf.float32)  # 0.5 를 기준으로 0과 1로 분류함
accuracy = tf.reduce_mean(tf.cast(tf.equal(predicted, Y), dtype=tf.float32))  # 트레인 데이터 Y 랑 예측한값이 얼마나 맞는지 정확도를 구해봄

# https://www.tensorflow.org/api_docs/python/tf/Session
with tf.Session() as s:  # Session 은 close 해줘야함...
    s.run(tf.global_variables_initializer())
    for step in range(10001):
        print(step, s.run([cost, train], feed_dict={X: x_data, Y: y_data}))

    print(s.run([hypothesis, predicted, accuracy], feed_dict={X: x_data, Y: y_data}))


