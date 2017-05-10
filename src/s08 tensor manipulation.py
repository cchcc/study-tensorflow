# encoding=utf8

import tensorflow as tf
import numpy as np

s = tf.Session()

arr = np.array([[1, 2, 3], [2, 2, 2], [10, 20, 30]])
print(arr.ndim)  # rank
print(arr.shape)

# rank 앞에 [ 이거 갯수
# shape (?,?,?) 일때 ? 가 3개 이므로 rank 가 3, 배열의 제일 안쪽부터 한차원씩 묶어가며 세봄
# axis 제일 바깥에꺼가 0 부터 시작, -1 은 제일 안쪽꺼

# broadcasting  행렬 연산시에 rank 나 shape 이 안맞아도 연산이 되도록 만들어 주는것

a1 = tf.constant([[1, 2], [3, 4]])
a2 = tf.constant([10])

# tf.matmul(a1, a2).eval()
print((a1 + a2).eval(session=s))

# reduce_mean 쓸때는 float 형 쓰는거 주의, 보통 -1로 제일 안쪽에걸 사용
# reduce_sum
# argmax

print(tf.reduce_mean([1,2]).eval(session=s))
print(tf.reduce_mean([[1.,2.,3.],[10.,20.,30.]]).eval(session=s))
print(tf.reduce_mean([[1.,2.,3.],[10.,20.,30.]],axis=-1).eval(session=s))


# reshape 대체로 가장 안쪽값은 그대로 놔둠

t = np.array([[1,2,3],[1,2,1]])
print(tf.reshape(t,[-1]).eval(session=s))
print(tf.reshape(t,[-1,2]).eval(session=s))

print(tf.squeeze([[0],[1]]).eval(session=s))
print(tf.squeeze(t).eval(session=s))


print(tf.ones_like(t).eval(session=s))  # shape 은 같고 1로 채움
print(tf.zeros_like(t).eval(session=s))  # shape 은 같고 0으로 채움