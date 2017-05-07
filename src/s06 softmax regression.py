# encoding=utf8

# multinomial
# 예측을 해야되는 결과가 여러개라면(A,B,C 중 하나)?? 하나하나를 클래스라고함
# binary logistic classification 을 각각 하나씩 적용해서 생각해보자(A 인지 아닌지 & B 인지 아닌지 & C 인지 아닌지)
# 각각 독립적으로 계산하면 구현등이 번거로우니 행렬곱을 이용하자

# softmax
# 각각 결과가 나왔을에 결과 합이 1 로 나오게끔 각 클래스의 확률을 계산함
# 그후에 그중에서 어떤건지 max 값으로 결과를 도출함

# cross entropy cost function
# 각 결과의 분배된 값과 실제값을 비교함

import tensorflow as tf

x_data = [[1,2,1,1],[2,1,3,2],[3,1,3,4],[4,1,5,5],[1,7,5,5],[1,2,5,6],[1,6,6,6],[1,7,7,7]]
y_data = [[0,0,1],[0,0,1],[0,0,1],[0,1,0],[0,1,0],[0,1,0],[1,0,0],[1,0,0]]
# one-hot encoding 선택지 중에 하나만 hot 하다. 맞는거는 1 아닌거는 모두 0 으로 표시함

X = tf.placeholder('float', [None, 4])
Y = tf.placeholder('float', [None, 3])  # 결과 label 의 갯수

nb_classes = 3

W = tf.Variable(tf.random_normal([4, nb_classes]))
b = tf.Variable(tf.random_normal([nb_classes]))

hypothesis = tf.nn.softmax(tf.matmul(X, W) + b)

cost = tf.reduce_mean(-tf.reduce_mean(Y * tf.log(hypothesis), axis=1))  # axis 는 몇차원 배열인지를 지정함
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.1).minimize(cost)

with tf.Session() as s:
    s.run(tf.global_variables_initializer())
    # print(s.run(tf.reduce_mean([[1.,1.,1.],[4.,4.,4.]])))
    # print(s.run(tf.reduce_mean([[1.,1.,1.],[4.,4.,4.]], axis=1)))

    for step in range(2001):
        print(step, s.run([optimizer, cost], feed_dict={X: x_data, Y: y_data}))

    testcase = s.run(hypothesis, feed_dict={X: [[1, 2, 1, 1]]})  # 결과가 이렇게 나왔을때
    print('testcase', testcase, s.run(tf.arg_max(testcase,1)))  # 그중에서 제일 비중이 큰놈(max)이 몇번째 인지 인덱스를 찾아냄

    r_x_data = s.run(hypothesis, feed_dict={X: x_data})
    print('x_data', r_x_data, s.run(tf.arg_max(r_x_data,1)))  # arg_max 의 1 은 몇차원 배열인지(0이 1차원 1이 2차원)