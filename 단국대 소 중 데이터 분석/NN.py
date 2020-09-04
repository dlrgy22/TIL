import tensorflow as tf
import pandas as pd
import numpy as np
from sklearn import utils


def get_train_data():
    df = pd.read_csv('train.csv')
    head = df.columns.values
    test_cel = df.values[:1000, 1:-1]
    test_class = df.values[:1000, -1]
    celestial_data = df.values[1000:, 1:-1]
    Classification_data = df.values[1000:, -1]

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

W1 = tf.get_variable("W1", shape=[18, 18 * 15], initializer=tf.contrib.layers.xavier_initializer())
b1 = tf.Variable(tf.random_normal([18 * 15]))
L1 = tf.nn.relu(tf.matmul(celestial, W1) + b1)
L1 = tf.nn.dropout(L1, keep_prob=keep_prob)

W2 = tf.get_variable("W2", shape=[18 * 15, 18 * 15], initializer=tf.contrib.layers.xavier_initializer())
b2 = tf.Variable(tf.random_normal([18 * 15]))
L2 = tf.nn.relu(tf.matmul(L1, W2) + b2)
L2 = tf.nn.dropout(L2, keep_prob=keep_prob)

W3 = tf.get_variable("W3", shape=[18 * 15, 18 * 15], initializer=tf.contrib.layers.xavier_initializer())
b3 = tf.Variable(tf.random_normal([18 * 15]))
L3 = tf.nn.relu(tf.matmul(L2, W3) + b3)
L3 = tf.nn.dropout(L3, keep_prob=keep_prob)

W4 = tf.get_variable("W4", shape=[18 * 15, 18 * 15], initializer=tf.contrib.layers.xavier_initializer())
b4 = tf.Variable(tf.random_normal([18 * 15]))
L4 = tf.nn.relu(tf.matmul(L3, W4) + b4)
L4 = tf.nn.dropout(L4, keep_prob=keep_prob)

W5 = tf.get_variable("W5", shape=[18 * 15, 3], initializer=tf.contrib.layers.xavier_initializer())
b5 = tf.Variable(tf.random_normal([3]))
H = tf.matmul(L4, W5) + b5
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=H, labels=classification_one_hot))
#optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.01).minimize(cost)
optimizer = tf.train.AdamOptimizer(learning_rate=0.0001).minimize(cost)
prediction = tf.argmax(H, 1)
correct_prediction = tf.equal(tf.argmax(H, 1), tf.argmax(classification_one_hot, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

epochs = 100
batch_size = 500
total_batch = int(len(celestial_data) / batch_size)
shuffle = True
test_cel = (test_cel - mean) / std

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for epoch in range(epochs):
        avg_c = 0
        if shuffle:
            utils.shuffle(celestial_data, classification_data)
        for i in range(batch_size):
            start = i * batch_size
            end = start + batch_size

            cel_batch = celestial_data[start:end]
            class_batch = classification_data[start:end]
            feed_dict = {celestial: cel_batch, classification: class_batch, keep_prob: 0.5}

            c, _ = sess.run([cost, optimizer], feed_dict=feed_dict)
            avg_c += c / total_batch
        print('Epoch:', '%04d' % (epoch + 1), 'cost =', '{:.9f}'.format(avg_c))
        acc = sess.run(accuracy, feed_dict={celestial: test_cel, classification: test_class, keep_prob: 1})
        print(acc)

    print('Learning Finished!')

    acc = sess.run(accuracy, feed_dict={celestial: test_cel, classification: test_class, keep_prob: 1})
    print(acc)

    test_data = get_test_data()
    test_data = (test_data - mean) / std
    pred = sess.run(prediction, feed_dict={celestial: test_data, keep_prob : 1})
    make_submission(pred)