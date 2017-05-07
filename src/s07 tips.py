# encoding=utf8

# learning rate 를 어떻게 잘 정할수 있을까?
# learning rate 값이 너무 커서 그래프 위쪽으로 가거나 최소값이 잘 안잡힘(overshooting)
# 너무작으면 시간이 오래걸리고 step 수가 최대로 갈때까지 최적화가 안될수 있다
# cost 값이 정상적을 잘 줄어드는지 확인하자!

# data preprocessing
# gradient descent 가 잘 적용되기 위해 인풋 데이터를 선처리 필요한경우가 있다
# 데이터 값의 분포 차이가 너무 큰경우(중간에 뭔가 데이터가 이상하게 크거나 작은게 있는경우) 데이터를 normalize 한다고 한다 (standardization)
# 데이터 전체를 0,0 으로 옮겨주거나 w,h 를 똑같도록 맞춰본다
# xy = MinMaxScaler(xy)

# overfitting 모델을 대략적으로 맞춘게 아니라 너무 딱맞춰서 더 복잡하게 만드는것(곡선을 너무 구부리도록 만듬)
# 이렇게 안되도록 하려면? train data 를 많이 만듬
# regularization 적용. 그래프 곡선을 좀더 완화 시키는것 l2reg = 0.001 * tf.reduce_sum(tf.square(W))
# 0.001 은 특정상수(람다), regularization strength

# 내가 만든 모델이 제대로 만들어진걸까 어떻게 평가할까? train set 을 적용해서 정확도를 측정하는 방식은 소용이 없다
# 데이터양을 7:3 으로 나누어서 train 7, test 3 으로 나눠서 train set 으로 모델을 만들고 test set 으로 평가해본다
# train set 을 다시 train, validation 으로 다시 나누고 validation 은 learning rate 와 regularization 상수를 구하는데 사용한다.

# online learning 전체 데이터를 특정 양으로 나눠서 학습 시킴. 나중에 새로운 데이터가 그만큼 생기면 그때또 학습시킴


# MNIST dataset - 숫자 손글씨 인식

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data  # 없으면 최초 다운로드 받아옴
mnist = input_data.read_data_sets('MNIST_data/', one_hot=True)

nb_classes = 10  # 숫자는 0~9 까지 10개

X = tf.placeholder(tf.float32, [None, 784])  # 이미지 픽셀이 28 * 28
Y = tf.placeholder(tf.float32, [None, nb_classes])

W = tf.Variable(tf.random_normal([784, nb_classes]))
b = tf.Variable(tf.random_normal([nb_classes]))

hypothesis = tf.nn.softmax(tf.matmul(X, W) + b)

cost = tf.reduce_mean(-tf.reduce_sum(Y * tf.log(hypothesis), axis=1))  # softmax cross entropy
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.1).minimize(cost)

predict_result = tf.equal(tf.arg_max(hypothesis, 1), tf.arg_max(Y, 1))
accuracy = tf.reduce_mean(tf.cast(predict_result, tf.float32))

epochs = 15  # 전체 데이터가 한번 돌면서 학습시킨건이 1 epoch . 똑같은걸 여러번 돌리면
batch_size = 100  # 전체 데이터를 몇개로 나누어서 학습할건지 (왜냐면 메모리 문제로)

with tf.Session() as s:
    s.run(tf.global_variables_initializer())

    for e in range(epochs):  # epochs 만큼 돌릴거임
        avg_cost = 0
        total_batch = int(mnist.train.num_examples / batch_size)  # 한 epoch 에서 몇개씩 몇번 돌릴건지

        for i in range(total_batch):  # total_batch 가 iteration
            xs, ys = mnist.train.next_batch(batch_size)
            c, _ = s.run([cost, optimizer], feed_dict={X: xs, Y: ys})
            print(avg_cost, c, total_batch)
            avg_cost += c / total_batch

        print('epoch %d' % e, 'cost %f' % avg_cost)

    # 어떤 tensor 하나만 돌리고자 할때 eval 을 사용함
    print('accuracy %f' % accuracy.eval(session=s, feed_dict={X: mnist.test.images, Y: mnist.test.labels}))  # 정확도는 test set 으로