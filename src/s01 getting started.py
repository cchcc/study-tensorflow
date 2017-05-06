# coding=utf-8
import tensorflow as tf

print("tensorflow version : " + tf.__version__)

# tensor : 텐서플로우에서 사용하는 데이터의 기본 구성 단위
node1 = tf.constant(3.0, tf.float32)  # 그냥 상수
node2 = tf.constant(4.0)
hellotf = tf.constant("hello tensorflow")
print(node1, node2)

# rank 몇차원 배열인지
# shape 몇차원 배열에서 안에 구성 요소가 몇개씩 들어있는지 배열의 제일 안쪽에 있는거부터 뒤에다가 표현

s = tf.Session()  # session : 런타임의 제어와 상태를 캡슐화함
print(s.run([node1, node2]))  # 뭐든 다 run 을 해야함
print(s.run(hellotf))

node3 = tf.add(node1, node2)
print(s.run(node3))

ph1 = tf.placeholder(tf.float32)  # placeholder : 나중에 값을 넣을 일종의 파라매터
ph2 = tf.placeholder(tf.float32)
adder_ph = ph1 + ph2
print(s.run(adder_ph, {ph1: 1.3, ph2: 2.2}))

W = tf.Variable([.3], tf.float32)  # variable : 모델을 만들기 위헤 train 가능한 파래매터. tensorflow 가 학습과정에서 변경시키는 값
b = tf.Variable([3.3], tf.float32)
x = tf.placeholder(tf.float32)
linear_model = W * x + b
s.run(tf.global_variables_initializer())  # variable 은 이거로 초기화해야함
print(s.run(linear_model, {x: [1, 2, 3, 4]}))

# tensorflow 동작 순서
# 1. tf operation 으로 그래프를 만든다
# 2. 데이터를 제공하고 그래프를 실행한다
# 3. 그래프 내에 필요한 variable 을 업뎃하고 결과를 받는다
