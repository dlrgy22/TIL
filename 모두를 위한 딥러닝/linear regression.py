import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
X = tf.placeholder(tf.float32)
Y = tf.placeholder(tf.float32)
test = [5, 6]

W = tf.Variable(tf.compat.v1.random_normal([1]), name='weight')
b = tf.Variable(tf.compat.v1.random_normal([1]), name='bias')

H = X * W + b


cost = tf.reduce_mean(tf.square(H - Y))               # 평균을 내준다

optimizer = tf.train.GradientDescentOptimizer(learning_rate = 0.01)
train = optimizer.minimize(cost)

sess = tf.Session()
sess.run(tf.global_variables_initializer())                # 변수 initalizer

for step in range(2001):
    cost_val, W_val, b_val, _ = sess.run([cost, W, b, train], feed_dict = {X :[1, 2, 3], Y : [2, 4, 6]})
    if step % 20 == 0:
        print(step, cost_val, W_val, b_val)

print(sess.run(H, feed_dict = {X : test}))