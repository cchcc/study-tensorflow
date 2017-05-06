# coding=utf8
import tensorflow as tf

x_data = [1, 2, 3]
y_data = [1, 2, 3]

W = tf.Variable(5.)
X = tf.placeholder(tf.float32)
Y = tf.placeholder(tf.float32)

hypothesis = X * W

cost = tf.reduce_mean(tf.square(hypothesis - Y))


# gradient descent algorithm
# cost funcion의 값을 작게 만드는 알고리즘
# 산꼭대기에서 밑으로 내려오는것과 같은 원리
# 현재 W 지점을 미분하여 그 경사각(기울기)이 작은쪽으로 점점 이동함
# 한번에 얼마만큼 감소시킬건지 -> learning rate

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

# convex function   볼록한 모양 - 어느지점에서 시작하더라도 최소값이 같게 나옴
# 3차원으로 생각했을때 다른지점에서 시작해보면 최소값이 다르게 나올수 있는데
# cost function 을 설계할때 그럴 가능성이 있는지 항상 확인해야함
# 아니면 gradient descent algorithm 적용 불가

# 머신러닝은 이론적으로 다음과 같은 과정을 거침
# 가설을 만듬(hypothesis) -> 가설이 적절한지 평가하는 방법을 만듬(cost function) -> cost function 을 기반으로 가설을 최적화 시키는 방법을 찾아내고 적용시킴 (gradient descent)