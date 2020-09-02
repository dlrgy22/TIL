import tensorflow as tf


x_data =[[1, 2], [2, 3], [3, 1], [4, 3], [5, 3], [6, 2]]
y_data = [[0], [0], [0], [1], [1], [1]]

X = tf.placeholder(tf.float32, shape = [None, 2])
Y = tf.placeholder(tf.float32, shape = [None, 1])
W = tf.Variable(tf.random_normal([2, 1]), name = 'weight')
b = tf.Variable(tf.random_normal([1]), name = 'bias')
H = tf.sigmoid(tf.matmul(X, W) + b)
cost = - tf.reduce_mean(Y * tf.log(H) + (1 - Y) * tf.log(1 - H))
train = tf.train.GradientDescentOptimizer(learning_rate = 0.05).minimize(cost)
predict = tf.cast(H > 0.5, dtype = tf.float32)
accuracy = tf.reduce_mean(tf.cast(tf.equal(predict, Y), dtype = tf.float32))

sess = tf.Session()
sess.run(tf.global_variables_initializer())

for step in range(10001):
    cost_val, _ = sess.run([cost, train], feed_dict = {X : x_data, Y : y_data})
    if step % 200 == 0:
        print(step, cost_val)

h, c, a = sess.run([H, predict, accuracy], feed_dict = {X : x_data, Y : y_data})
print("\n Hypothesis : ",h, "\n correct : ",c, "\n Accuacy : ", a)