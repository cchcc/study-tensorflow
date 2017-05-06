# coding=utf8
import tensorflow as tf

x_data = [1, 2, 3]
y_data = [1, 2, 3]

W = tf.Variable(5.)
X = tf.placeholder(tf.float32)
Y = tf.placeholder(tf.float32)

hypothesis = X * W

cost = tf.reduce_mean(tf.square(hypothesis - Y))

# Gradient Descent 를 직접 구현해봄 - W 값을 조정하기위해 W 에 Cost Function 미분한것(기울기)을 빼준다
gradient = tf.reduce_mean(((W * X - Y) * X))
# learning_rate = 0.1
# descent = W - learning_rate * gradient
# update = W.assign(descent)

# Gradient Descent
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.01)
# train = optimizer.minimize(cost)
# 최소화 시키는 부분을 다른방법으로 처리해봄
gvs = optimizer.compute_gradients(cost)
# 여기서 필요하다면 gvs 를 다르게 적용해서 알고리즘을 약간 변경해볼수 있음
apply_gradients = optimizer.apply_gradients(gvs)

s = tf.Session()
s.run(tf.global_variables_initializer())

for step in range(100):
    # s.run(update, feed_dict={X: x_data, Y: y_data})
    # print(step, s.run(cost, feed_dict={X: x_data, Y: y_data}), s.run(W))
    print(step, s.run([gradient, W, gvs], feed_dict={X: x_data, Y: y_data}))
    s.run(apply_gradients, feed_dict={X: x_data, Y: y_data})

# 가설을 만듬(hypothesis) -> 가설이 적절한지 평가하는 방법을 만듬(cost function) -> cost function 을 기반으로 가설을 최적화 시킴(gradient descent)