# coding=utf8
import tensorflow as tf

# 1 그래프 만들기
# x_train = [1, 2, 3]  # supervised 된 데이터 셋
# y_train = [1, 2, 3]
X = tf.placeholder(tf.float32, shape=[None])  # 위에걸 placeholder 로 나중에 값을 대입할수 있도록
Y = tf.placeholder(tf.float32, shape=[None])  # shape 은 1차원 배열에 아무값이나 들어 올수 있다는 의미

W = tf.Variable(tf.random_normal([1]), name='weight')  # random_normal 로 랜덤으로 값을주면서 shape 이 [1] 로 지정
b = tf.Variable(tf.random_normal([1]), name='bias')
# hypothesis = x_train * W + b
hypothesis = X * W + b

# cost function
# cost = tf.reduce_mean(tf.square(hypothesis - y_train))  # reduce_mean 은 텐서의 값들을 평균을 구해줌
cost = tf.reduce_mean(tf.square(hypothesis - Y))

# cost function 의 값을 Gradient Descent 라는 방법으로 최소화 시킴
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.01)  # learning_rate 는 얼마큼씩 감소 시킬건지
train = optimizer.minimize(cost)

# 2 만든그래프에 값들을 세팅하고 돌려봄
s = tf.Session()
s.run(tf.global_variables_initializer())

for step in range(2001):  # 2000 번을 돌려봄
    # s.run(train)  # train 을 돌리면 여기에 물려있는게 cost 이고 cost 에 물려있는게 hypothesis, 거기에 W,b 이런식으로 그래프가 그려짐
    # print(step, s.run(cost), s.run(W), s.run(b))
    print(step, s.run([cost, W, b, train], feed_dict={X: [1, 2, 3, 4, 5], Y: [1, 2, 3, 4, 5]}))

# 에측해볼값을 한번 넣어봄
print(s.run(hypothesis, feed_dict={X: [10]}))
