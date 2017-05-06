# encoding=utf8
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
import tensorflow as tf

# https://www.tensorflow.org/programmers_guide/reading_data
# queue runners 데이터를 효율적으로 불러오는 시스템을 제공함
# 1. 읽어올 파일들 지정
file_queue = tf.train.string_input_producer(['s04-2_test_score.csv'], shuffle=False)

# 2. 리더지정
reader = tf.TextLineReader()
k,v = reader.read(file_queue)

# 3. 파싱 방법 지정
xy = tf.decode_csv(v, record_defaults=[[0.],[0.],[0.],[0.]])  # record_defaults 이걸로 이게 무슨 타입인지 어떤 형태인지 지정

# 4.
x_data_batch, y_data_batch = tf.train.batch([xy[0:-1], xy[-1:]], batch_size=10)  # batch_size 는 한번에 몇개씩 가져올것인지 지정

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

coord = tf.train.Coordinator()
threads = tf.train.start_queue_runners(sess=s,coord=coord)

for step in range(2001):
    # print(s.run([cost, hypothesis, train], feed_dict={x1:x1_data, x2:x2_data, x3:x3_data, Y:y_data}))
    x_data, y_data = s.run([x_data_batch, y_data_batch])
    s.run(train, feed_dict={X:x_data, Y:y_data})
    print(step, s.run([W, b]))


coord.request_stop()
coord.join(threads)

print(s.run(hypothesis, feed_dict={X:[[73,80,75]]}))
print(s.run(hypothesis, feed_dict={X:[[93,88,93]]}))

# epoch 트레이닝 셋 전체를 한번 돌렸을때 한 epoch(세대) 를 했다고함