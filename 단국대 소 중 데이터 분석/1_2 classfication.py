import tensorflow as tf
import pandas as pd
import numpy as np
from imblearn.over_sampling import *

def make_trainsubmission(pred):
    head = ['id', 'class']
    data = []
    for i in range(len(pred)):
        data.append([i, pred[i]])
    data = np.vstack((head, data))
    file_name = "trainsubmission.csv"
    np.savetxt(file_name, data, fmt="%s", delimiter=",")

def get_train_data():
    df = pd.read_csv('train.csv')
    head = df.columns.values
    test_cel = df.values[:1000, 1:-1]
    test_class = df.values[:1000, -1]
    celestial_data = df.values[1000:, 1:-1]
    Classification_data = df.values[1000:, -1]

    return test_cel, test_class, celestial_data, Classification_data

def get_trainsubmission():
    df = pd.read_csv('trainsubmission.csv')
    train_s = df.values[:,-1]
    return train_s

def get_submission():
    df = pd.read_csv('submission.csv')
    train_m = df.values[:,-1]
    return train_m

def one_two(celestial_data, Classification_data):
    ot_cel = []
    ot_class = []
    for i in range(len(Classification_data)):
        if Classification_data[i] != 0:
            ot_cel.append(celestial_data[i])
            ot_class.append(Classification_data[i] - 1)
    return np.array(ot_cel), np.array(ot_class)

test_cel, test_class, celestial_data, Classification_data = get_train_data()
train_s = get_trainsubmission()

# zero = 0
# one = 0
# two = 0
#
# for i in range(len(train_s)):
#     if train_s[i] != Classification_data[i]:
#         if train_s[i] == 0:
#             zero += 1
#         elif train_s[i] == 1:
#             one += 1
#         else:
#             two += 1
#
# print(zero, one, two)

train_m = get_submission()

ot_cel, ot_class = one_two(celestial_data, Classification_data)
#X_samp, y_samp = SMOTE(random_state=4).fit_sample(ot_cel, ot_class)
#s = np.arange(X_samp.shape[0])
# np.random.shuffle(s)
# X_samp = X_samp[s]
# y_samp = y_samp[s]
# print(len(y_samp), len(X_samp))
# mean = X_samp.mean(axis=0)
# std = X_samp.std(axis=0)
# ot_cel = (ot_cel - mean) / std
mean = ot_cel.mean(axis=0)
std = ot_cel.std(axis=0)
ot_cel = (ot_cel - mean) / std
nb_classes = 2

celestial = tf.placeholder(tf.float32, [None, 18])
classification = tf.placeholder(tf.int32, [None])
classification_one_hot = tf.one_hot(classification, nb_classes)
classification_one_hot = tf.reshape(classification_one_hot, [-1, nb_classes])
time = [1000]
result = []
#keep_prob = tf.placeholder(tf.float32)
size = 18
W1 = tf.get_variable("W1", shape=[18, size], initializer=tf.contrib.layers.variance_scaling_initializer())
b1 = tf.Variable(tf.random_normal([size]))
L1 = tf.nn.relu(tf.matmul(celestial, W1) + b1)
#L1 = tf.nn.dropout(L1, keep_prob=keep_prob)

W3 = tf.get_variable("W5", shape=[size, 2], initializer=tf.contrib.layers.xavier_initializer())
b3 = tf.Variable(tf.random_normal([2]))
H = tf.matmul(L1,W3) + b3
cost = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=H, labels=classification_one_hot))

optimizer = tf.train.AdamOptimizer(learning_rate=0.01).minimize(cost)
prediction = tf.argmax(H, 1)
correct_prediction = tf.equal(tf.argmax(H, 1), tf.argmax(classification_one_hot, 1) + 1)
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for epoch in range(9650):
        # s = np.arange(X_samp.shape[0])
        # np.random.shuffle(s)
        # X_samp = X_samp[s]
        # y_samp = y_samp[s]

        feed_dict = {celestial: ot_cel, classification: ot_class}
        #feed_dict = {celestial: X_samp, classification: y_samp}
        c, _ = sess.run([cost, optimizer], feed_dict=feed_dict)

        print('Epoch:', '%04d' % (epoch + 1), 'cost =', '{:.9f}'.format(c))
    pred = sess.run(prediction, feed_dict={celestial: ot_cel})
    c = 0
    for i in range(len(pred)):
        if pred[i] == ot_class[i]:
            c += 1
    print(c/ len(ot_class))
    for i in range(len(train_s)):
        if train_s[i] != 0:
            input = celestial_data[i].reshape(-1,18)
            pred = sess.run(prediction, feed_dict={celestial: input})
            train_s[i] = pred + 1
    count = 0
    zero = 0
    one = 0
    two = 0
    for i in range(len(train_s)):
        if train_s[i] == Classification_data[i]:
            count += 1
        else:
            if Classification_data[i] == 0:
                zero += 1
            elif Classification_data[i] == 1:
                one += 1
            else:
                two +=1

    print(zero, one, two)
    print(count / len(train_s))

    # for i in range(len(train_m)):
    #     if train_m[i] != 2:
    #         input = celestial_data[i].reshape(-1,18)
    #         pred = sess.run(prediction, feed_dict={celestial: input})
    #         train_m[i] = pred




