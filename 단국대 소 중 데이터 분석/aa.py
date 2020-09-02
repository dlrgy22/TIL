import pandas as pd
import tensorflow as tf
from sklearn.tree import DecisionTreeClassifier
import numpy as np
train = pd.read_csv('train.csv', index_col=0)
test = pd.read_csv('test.csv', index_col=0)
sample_submission = pd.read_csv('sample_submission.csv', index_col=0)
train_x = train.drop(columns='class', axis=1) # class 열을 삭제한 새로운 객체
train_y = train['class'] # 결과 레이블(class)
test_x = test

mean = train_x.mean(axis=0)
std = train_x.std(axis=0)

train_x = (train_x - mean) / std
test_x = (test_x - mean) / std
nb_classes = 3
print(test_x.shape)
X = tf.placeholder(tf.float32, [None, 18])
Y = tf.placeholder(tf.int32, [None])
Y_one_hot = tf.one_hot(Y, nb_classes)
Y_one_hot = tf.reshape(Y_one_hot, [-1, nb_classes])
W = tf.get_variable("weight", shape=[18, nb_classes], initializer=tf.contrib.layers.xavier_initializer())
# W = tf.Variable(tf.random_normal([18, nb_classes]), name="weight")
b = tf.Variable(tf.random_normal([nb_classes]), name="bias")
logits = tf.matmul(X, W) + b
hypothesis = tf.nn.softmax(logits)
cost_i = tf.nn.softmax_cross_entropy_with_logits_v2(logits=logits, labels=Y_one_hot) # 교차 엔트로피 오차
cost = tf.reduce_mean(cost_i)
optimizer = tf.train.AdamOptimizer(learning_rate=0.1).minimize(cost)
prediction = tf.argmax(hypothesis, 1) # 예측값, 각 행에서 가장 큰 값의 인덱스를 추출, argmax( , 0)이면 열에서 가장 큰 값의 인덱스
correct_prediction = tf.equal(prediction, tf.argmax(Y_one_hot, 1)) # 예측값과 정답이 얼마나 일치하는가
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32)) # 정확도

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for step in range(2000):
        sess.run(optimizer, feed_dict={X: train_x, Y: train_y})
        if step % 100 == 0:
            loss, acc = sess.run([cost, accuracy], feed_dict={X: train_x, Y: train_y})
            print("Step: {:5}\tLoss: {:.3f}\tAcc: {:.2f}".format(step, loss, acc))