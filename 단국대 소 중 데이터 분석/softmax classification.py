import tensorflow.compat.v1 as tf
import pandas as pd
import numpy as np

tf.compat.v1.disable_eager_execution()

def get_train_data():
    df = pd.read_csv('train.csv')
    head = df.columns.values
    celestial_data = df.values[:,1:-1]
    Classification_data = df.values[:,-1]

    return celestial_data, Classification_data

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

celestial_data, classification_data = get_train_data()
print(celestial_data[1])
print(celestial_data[1][1])
nb_classes = 3
celestial = tf.placeholder(tf.float32, [None, 18])
classification = tf.placeholder(tf.int32, [None])
classification_one_hot = tf.one_hot(classification, nb_classes)
classification_one_hot = tf.reshape(classification_one_hot, [-1, nb_classes])

W = tf.Variable(tf.random_normal([18, nb_classes]), name = 'weight')
b = tf.Variable(tf.random_normal([nb_classes]), name = 'bias')

logit = tf.matmul(celestial, W) + b
H = tf.nn.softmax(logit)

cost_i = tf.nn.softmax_cross_entropy_with_logits(logits=logit, labels = classification_one_hot)
cost = tf.reduce_mean(cost_i)
optimizer = tf.train.GradientDescentOptimizer(learning_rate = 0.005).minimize(cost)
prediction = tf.argmax(H, 1)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    for step in range(2000):

        sess.run(optimizer, feed_dict = {celestial : celestial_data, classification : classification_data})
        if step % 100 == 0:
            print(step, sess.run(cost, feed_dict = {celestial : celestial_data, classification : classification_data}))


    pred = sess.run(prediction, feed_dict = {celestial : celestial_data})
    print(pred)

    test_data = get_test_data()
    pred = sess.run(prediction, feed_dict = {celestial : test_data})
    make_submission(pred)

