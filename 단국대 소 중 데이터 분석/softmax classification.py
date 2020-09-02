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

W = tf.get_variable("weight", shape=[18, nb_classes], initializer=tf.contrib.layers.xavier_initializer())
b = tf.Variable(tf.random_normal([nb_classes]), name = 'bias')

logit = tf.matmul(celestial, W) + b
H = tf.nn.softmax(logit)

cost_i = tf.nn.softmax_cross_entropy_with_logits(logits=logit, labels = classification_one_hot)
cost = tf.reduce_mean(cost_i)
optimizer = tf.train.AdamOptimizer(learning_rate=0.1).minimize(cost)
prediction = tf.argmax(H, 1)
correct_prediction = tf.equal(prediction, tf.argmax(classification_one_hot, 1)) # 예측값과 정답이 얼마나 일치하는가
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32)) # 정확도


with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    for step in range(2000):

        sess.run(optimizer, feed_dict = {celestial : celestial_data, classification : classification_data})
        if step % 100 == 0:
            loss, acc = sess.run([cost, accuracy], feed_dict = {celestial : celestial_data, classification : classification_data})
            print("Step: {:5}\tLoss: {:.3f}\tAcc: {:.2f}".format(step, loss, acc))


    # pred = sess.run(prediction, feed_dict = {celestial : celestial_data})
    # print(pred)

    test_cel = (test_cel - mean) / std
    acc = sess.run(accuracy, feed_dict = {celestial : celestial_data, classification : classification_data})
    print(acc)

    test_data = get_test_data()
    test_data = (test_data - mean) / std
    pred = sess.run(prediction, feed_dict = {celestial : test_data})
    make_submission(pred)

