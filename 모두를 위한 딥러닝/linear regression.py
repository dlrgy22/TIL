import tensorflow as tf
x_train = [1, 2, 3]
y_train = [1, 2, 3]
W = tf.Variable(tf.compat.v1.random_normal([1]), name='weight')
b = tf.Variable(tf.compat.v1.random_normal([1]), name='bias')

H = x_train * W + b

cost = tf.reduce_mean(tf.square(H - y_train))               # 평균을 내준다

optimizer = tf.compat.v1.train.GradientDescentOptimizer(learning_rate = 0.01)
train = optimizer.minimize(cost)

sess = tf.Session()
sess.run(tf.global)