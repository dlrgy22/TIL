import tensorflow as tf
import pandas as pd
import numpy as np
from imblearn.under_sampling import *
from imblearn.over_sampling import *
from sklearn import utils

tf.set_random_seed(777)  # reproducibility
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

# def std_scale(mean, std, celestial_data, test_cel):
#     for i in range(len(celestial_data)):
#         print(np.shape(celestial_data), np.shape(mean),  np.shape(std))
#         celestial_data[i] -= mean
#         celestial_data[i] /= std
#     for i in range(test_data):
#         test_data[i] -= mean
#         test_data[i] /= std
#     return celestial_data, test_data


test_cel, test_class, celestial_data, classification_data = get_train_data()
#X_samp, y_samp = RandomUnderSampler(random_state=0).fit_sample(celestial_data, classification_data)
#X_samp, y_samp = RandomOverSampler(random_state=0).fit_sample(celestial_data, classification_data)
#X_samp, y_samp = ADASYN(random_state=0).fit_sample(celestial_data, classification_data)
#X_samp, y_samp = SMOTE(random_state=4).fit_sample(celestial_data, classification_data)
#X_samp, y_samp = SMOTETomek(random_state=4).fit_sample(celestial_data, classification_data)

# zero = 0
# one = 0
# two = 0
# for i in range(len(y_samp)):
#     if y_samp[i] == 0:
#         zero += 1
#     elif y_samp[i] == 1:
#         one += 1
#     else:
#         two += 1
# print(zero, one, two)
# s = np.arange(X_samp.shape[0])
# np.random.shuffle(s)
# X_samp = X_samp[s]
# y_samp = y_samp[s]
# test_cel = X_samp[:1000]
# test_class = y_samp[:1000]
mean = celestial_data.mean(axis=0)
std = celestial_data.std(axis=0)
celestial_data = (celestial_data - mean) / std
test_cel = (test_cel - mean) / std
# X_samp = (X_samp - mean) / std


nb_classes = 3

celestial = tf.placeholder(tf.float32, [None, 18])
classification = tf.placeholder(tf.int32, [None])
classification_one_hot = tf.one_hot(classification, nb_classes)
classification_one_hot = tf.reshape(classification_one_hot, [-1, nb_classes])
time = [100000]
result = []
#keep_prob = tf.placeholder(tf.float32)
size = 36
W1 = tf.get_variable("W1", shape=[18, size], initializer=tf.contrib.layers.variance_scaling_initializer())
b1 = tf.Variable(tf.random_normal([size]))
L1 = tf.nn.relu(tf.matmul(celestial, W1) + b1)
#L1 = tf.nn.dropout(L1, keep_prob=keep_prob)

W2 = tf.get_variable("W2", shape=[size, 9], initializer=tf.contrib.layers.variance_scaling_initializer())
b2 = tf.Variable(tf.random_normal([9]))
L2 = tf.nn.relu(tf.matmul(L1, W2) + b2)
# L2 = tf.nn.dropout(L2, keep_prob=keep_prob)
#
# W3 = tf.get_variable("W3", shape=[size, size], initializer=tf.contrib.layers.variance_scaling_initializer())
# b3 = tf.Variable(tf.random_normal([size]))
# L3 = tf.nn.leaky_relu(tf.matmul(L2, W3) + b3)
# L3 = tf.nn.dropout(L3, keep_prob=keep_prob)
#
# W4 = tf.get_variable("W4", shape=[size, size], initializer=tf.contrib.layers.variance_scaling_initializer())
# b4 = tf.Variable(tf.random_normal([size]))
# L4 = tf.nn.leaky_relu(tf.matmul(L3, W4) + b4)
# L4 = tf.nn.dropout(L4, keep_prob=keep_prob)

W5 = tf.get_variable("W5", shape=[9, 3], initializer=tf.contrib.layers.xavier_initializer())
b5 = tf.Variable(tf.random_normal([3]))
H = tf.matmul(L2,W5) + b5
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=H, labels=classification_one_hot))
#optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.01).minimize(cost)
optimizer = tf.train.AdamOptimizer(learning_rate=0.01).minimize(cost)
prediction = tf.argmax(H, 1)
correct_prediction = tf.equal(tf.argmax(H, 1), tf.argmax(classification_one_hot, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
for t in time:
    epochs = t
    # s = np.arange(X_samp.shape[0])
    # np.random.shuffle(s)
    # X_samp = X_samp[s]
    # y_samp = y_samp[s]
    #batch_size = 32
    #total_batch = int(len(celestial_data) / batch_size)=

    # with tf.Session() as sess:
    #     sess.run(tf.global_variables_initializer())
    #     for epoch in range(epochs):
    #         avg_c = 0
    #         #if shuffle:
    #             #utils.shuffle(celestial_data, classification_data)
    #         for i in range(batch_size):
    #             start = i * batch_size
    #             end = start + batch_size
    #
    #             cel_batch = celestial_data[start:end]
    #             class_batch = classification_data[start:end]
    #             feed_dict = {celestial: cel_batch, classification: class_batch}
    #
    #             c, _ = sess.run([cost, optimizer], feed_dict=feed_dict)
    #             avg_c += c / total_batch
    #         acc = sess.run(accuracy, feed_dict={celestial: test_cel, classification: test_class})
    #         print('Epoch:', '%04d' % (epoch + 1), 'cost =', '{:.9f}'.format(avg_c))
    #         print(acc)
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        for epoch in range(epochs):


            feed_dict = {celestial: celestial_data, classification: classification_data}
            c, _ = sess.run([cost, optimizer], feed_dict=feed_dict)
            #acc = sess.run(accuracy, feed_dict={celestial: test_cel, classification: test_class})

            print('Epoch:', '%04d' % (epoch + 1), 'cost =', '{:.9f}'.format(c))
            #print(acc)

            if epoch % 10000 == 0:
                acc = sess.run(accuracy, feed_dict={celestial: test_cel, classification: test_class})
                train_acc = sess.run(accuracy, feed_dict={celestial: celestial_data, classification: classification_data})
                pred = sess.run(prediction, feed_dict={celestial: celestial_data})
                z_count = 0
                o_count = 0
                t_count = 0
                z_c = 0
                o_c = 0
                t_c = 0
                zero = 0
                one = 0
                two = 0
                for i in range(len(pred)):
                    if pred[i] == 0:
                        z_count += 1
                    elif pred[i] == 1:
                        o_count += 1
                    else:
                        t_count += 1

                    if pred[i] != classification_data[i]:
                        if classification_data[i] == 0:
                            zero += 1
                        elif classification_data[i] == 1:
                            one += 1
                        else:
                            two += 1
                    else:
                        if pred[i] == 0:
                            z_c += 1
                        elif pred[i] == 1:
                            o_c += 1
                        else:
                            t_c += 1
                result.append([epoch, c, acc, train_acc, z_c/z_count, o_c/o_count, t_c/t_count, zero, one, two])
                        # print(pred[i], classification_data[i])

        print('Learning Finished!')

        # acc = sess.run(accuracy, feed_dict={celestial: test_cel, classification: test_class})
        # train_acc = sess.run(accuracy, feed_dict={celestial: celestial_data, classification: classification_data})
        # result.append([t, c, acc, train_acc])
        for element in result:
            print(element[:4])
            print(element[4:7])
            print(element[7:])
        print(result)
        # pred = sess.run(prediction, feed_dict={celestial: celestial_data})
        # z_count =0
        # o_count = 0
        # t_count = 0
        # z_c = 0
        # o_c = 0
        # t_c = 0
        # zero = 0
        # one = 0
        # two = 0
        # for i in range(len(pred)):
        #     if pred[i] == 0:
        #         z_count += 1
        #     elif pred[i] == 1:
        #         o_count += 1
        #     else:
        #         t_count += 1
        #
        #     if pred[i] != classification_data[i]:
        #         if classification_data[i] == 0:
        #             zero += 1
        #         elif classification_data[i] == 1:
        #             one += 1
        #         else:
        #             two += 1
        #     else:
        #         if pred[i] == 0:
        #             z_c += 1
        #         elif pred[i] == 1:
        #             o_c += 1
        #         else:
        #             t_c += 1

                #print(pred[i], classification_data[i])
        print(z_c/z_count, o_c/o_count, t_c/t_count)
        print(zero, one, two)
        make_trainsubmission(pred)
        test_data = get_test_data()
        test_data = (test_data - mean) / std
        pred = sess.run(prediction, feed_dict={celestial: test_data})
        make_submission(pred)