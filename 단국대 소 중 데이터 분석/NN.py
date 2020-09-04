import tensorflow as tf
import pandas as pd
import numpy as np

def get_train_data():
    df = pd.read_csv('train.csv')
    head = df.columns.values
    test_cel = df.values[:100,1:-1]
    test_class = df.values[:100, -1]
    celestial_data = df.values[100:,1:-1]
    Classification_data = df.values[100:,-1]

    return test_cel, test_class, celestial_data, Classification_data

def get_test_data():
    df = pd.read_csv('test.csv')
    celestial_data = df.values[:, 1:]
    return celestial_data

def make_submission(classification):
    head = ['id', 'class']
    data = []
    for i in range(len(classification)):
        data.append([320000 + i, classification[i]])
    data = np.vstack((head, data))
    file_name = "submission.csv"
    np.savetxt(file_name, data, fmt="%s", delimiter=",")

test_cel, test_class, celestial_data, classification_data = get_train_data()
mean = celestial_data.mean(axis=0)
std = celestial_data.std(axis=0)
celestial_data = (celestial_data - mean) / std
nb_classes = 3


celestial = tf.placeholder(tf.float32, [None, 18])
classification = tf.placeholder(tf.int32, [None])
classification_one_hot = tf.one_hot(classification, nb_classes)
classification_one_hot = tf.reshape(classification_one_hot, [-1, nb_classes])

keep_prob = tf.placeholder(tf.float32)

W1 = tf.get_variable("W1", shape = [18, 18 * 2], initializer=tf.contrib.layers.xavier_initializer())
b1 = tf.Variable(tf.random_normal([18 * 2]))
L1 = tf.nn.relu(tf.matmul(celestial, W1) + b1)
L1 = tf.nn.dropout(L1, keep_prob = keep_prob)

W2 = tf.get_variable("W2", shape = [18 * 2, 18 * 2], initializer=tf.contrib.layers.xavier_initializer())
b2 = tf.Variable(tf.random_normal([18 * 2]))
L2 = tf.nn.relu(tf.matmul(L1, W2) + b2)
L2 = tf.nn.dropout(L2, keep_prob = keep_prob)

W3 = tf.get_variable("W3", shape = [18 * 2, 3], initializer=tf.contrib.layers.xavier_initializer())
b3 = tf.Variable(tf.random_normal([3]))
H = tf.matmul(L2, W3) + b3
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=H, labels=classification_one_hot))
optimizer = tf.train.AdamOptimizer(learning_rate=0.005).minimize(cost)

correct_prediction = tf.equal(tf.argmax(H, 1), tf.argmax(classification_one_hot, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for step in range(2000):
        feed_dict = {celestial : celestial_data , classification : classification_data , keep_prob: 0.5}
        c, _ = sess.run([cost, optimizer], feed_dict=feed_dict)
        if step % 100 == 0:
            print('step = ', step, 'cost =', '{:.9f}'.format(c))

    print('Learning Finished!')

    test_cel = (test_cel - mean) / std
    acc = sess.run(accuracy, feed_dict={celestial: test_cel, classification: test_class, keep_prob : 1})
    print(acc)