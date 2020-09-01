import tensorflow.compat.v1 as tf
import matplotlib.pyplot as plt

x_data = [1, 2, 3]
y_data = [1, 2, 3]
tf.compat.v1.disable_eager_execution()
X = tf.placeholder(tf.float32)
Y = tf.placeholder(tf.float32)
W = tf.Variable(tf.random_normal([1]), name = 'weight')
H = X * W
cost = tf.reduce_mean(tf.square(H - Y))

sess = tf.Session()
sess.run(tf.global_variables_initializer())

# W_val = []
# cost_val = []
#
# for i in range(-30, 50):
#     feed_W = i * 0.1
#     curr_cost, curr_W = sess.run([cost, W], feed_dict = {W : feed_W})
#     W_val.append(curr_W)
#     cost_val.append(curr_cost)
#
# plt.plot(W_val, cost_val)
# plt.show()

learning_rate = 0.1

gradient = tf.reduce_mean((W * X - Y) * X)
descent = W - learning_rate * gradient
update = W.assign(descent)

for step in range(21):
    sess.run(update, feed_dict = {X : x_data, Y : y_data})
    print(step, sess.run(cost, feed_dict = {X : x_data, Y : y_data}), sess.run(W))