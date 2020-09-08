import tensorflow as tf
import numpy as np
import pandas as pd
from sklearn import utils

tf.set_random_seed(777)  # reproducibility
nb_classes = 3

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

class Model:

    def __init__(self, sess, name, idx):
        self.sess = sess
        self.name = name
        self.idx = idx
        self._build_net()

    def _build_net(self):
        parm = [[tf.contrib.layers.variance_scaling_initializer(), 36, 0.001], [tf.contrib.layers.variance_scaling_initializer(), 54, 0.001], [tf.contrib.layers.variance_scaling_initializer(), 36, 0.001], [tf.contrib.layers.variance_scaling_initializer(), 54, 0.001], [tf.contrib.layers.variance_scaling_initializer(), 36, 0.001], [tf.contrib.layers.variance_scaling_initializer(), 54, 0.001], [tf.contrib.layers.xavier_initializer(), 36, 0.001], [tf.contrib.layers.xavier_initializer(), 54, 0.001]]

        with tf.variable_scope(self.name):
            self.training = tf.placeholder(tf.bool)

            self.celestial = tf.placeholder(tf.float32, [None, 18])
            self.classification = tf.placeholder(tf.int32, [None])
            self.classification_one_hot = tf.one_hot(self.classification, nb_classes)
            self.classification_one_hot = tf.reshape(self.classification_one_hot, [-1, nb_classes])

            self.keep_prob = tf.placeholder(tf.float32)

            size = parm[self.idx][1]
            W1 = tf.get_variable("W1", shape=[18, size], initializer=parm[self.idx][0])
            b1 = tf.Variable(tf.random_normal([size]))

            if self.idx == 0 or 1:
                L1 = tf.nn.relu(tf.matmul(self.celestial, W1) + b1)
            elif self.idx == 2 or 3:
                L1 = tf.nn.leaky_relu(tf.matmul(self.celestial, W1) + b1)
            elif self.idx == 4 or 5:
                L1 = tf.nn.elu(tf.matmul(self.celestial, W1) + b1)
            else:
                L1 = tf.sigmoid(tf.matmul(self.celestial, W1) + b1)

            L1 = tf.nn.dropout(L1, keep_prob=self.keep_prob)
            W2 = tf.get_variable("W2", shape=[size, size], initializer=parm[self.idx][0])
            b2 = tf.Variable(tf.random_normal([size]))

            if self.idx == 0 or 1:
                L2 = tf.nn.relu(tf.matmul(L1, W2) + b2)
            elif self.idx == 2 or 3:
                L2 = tf.nn.leaky_relu(tf.matmul(L1, W2) + b2)
            elif self.idx == 4 or 5:
                L2 = tf.nn.elu(tf.matmul(L1, W2) + b2)
            else:
                L2 = tf.nn.sigmoid(tf.matmul(L1, W2) + b2)

            L2 = tf.nn.dropout(L2, keep_prob=self.keep_prob)

            # W3 = tf.get_variable("W3", shape=[size, size], initializer=tf.contrib.layers.xavier_initializer())
            # b3 = tf.Variable(tf.random_normal([size]))
            # L3 = tf.nn.relu(tf.matmul(L2, W3) + b3)
            # L3 = tf.nn.dropout(L3, keep_prob=self.keep_prob)
            #
            # W4 = tf.get_variable("W4", shape=[size, size], initializer=tf.contrib.layers.xavier_initializer())
            # b4 = tf.Variable(tf.random_normal([size]))
            # L4 = tf.nn.relu(tf.matmul(L3, W4) + b4)
            # L4 = tf.nn.dropout(L4, keep_prob=self.keep_prob)

            W5 = tf.get_variable("W5", shape=[size, 3], initializer=parm[self.idx][0])
            b5 = tf.Variable(tf.random_normal([3]))
            H = tf.matmul(L2, W5) + b5

        self.cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=H, labels=self.classification_one_hot))
        self.optimizer = tf.train.AdamOptimizer(learning_rate=parm[self.idx][2]).minimize(self.cost)
        self.soft = tf.nn.softmax(H)
        self.prediction = tf.argmax(H, 1)
        correct_prediction = tf.equal(tf.argmax(H, 1), tf.argmax(self.classification_one_hot, 1))
        self.accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    def softmax(self, test_cel):
        return self.sess.run(self.soft, feed_dict={self.celestial: test_cel, self.keep_prob : 1})

    def predict(self, test_cel):
        return self.sess.run(self.prediction, feed_dict={self.celestial: test_cel, self.keep_prob : 1})

    def get_accuarcy(self, test_cel, test_class):
        return self.sess.run(self.accuracy, feed_dict={self.celestial: test_cel, self.classification: test_class, self.keep_prob: 1})

    def train(self, cel_batch, class_batch):
        return self.sess.run([self.cost, self.optimizer], feed_dict={self.celestial: cel_batch, self.classification: class_batch, self.keep_prob: 0.7})



test_cel, test_class, celestial_data, classification_data = get_train_data()
mean = celestial_data.mean(axis=0)
std = celestial_data.std(axis=0)
celestial_data = (celestial_data - mean) / std
test_cel = (test_cel - mean) / std
epochs = 500
batch_size = 100
total_batch = int(len(celestial_data) / batch_size)
shuffle = True

sess = tf.Session()

models = []
num_models = 8
for m in range(num_models):
    models.append(Model(sess, "model" + str(m), m))
sess.run(tf.global_variables_initializer())

print("Learning start !")

for epoch in range(epochs):
    avg_cost_list = np.zeros(len(models))
    acc_list = np.zeros(len(models))
    if shuffle:
        utils.shuffle(celestial_data, classification_data)
    for i in range(batch_size):
        start = i * batch_size
        end = start + batch_size

        cel_batch = celestial_data[start:end]
        class_batch = classification_data[start:end]
        for m_idx, m in enumerate(models):
            c, _ = m.train(cel_batch, class_batch)
            acc = m.get_accuarcy(test_cel, test_class)
            avg_cost_list[m_idx] += c / total_batch
            acc_list[m_idx] += acc


    print('Epoch:', '%04d' % (epoch + 1), 'cost =',avg_cost_list , 'acc = ', acc_list)
print("Learning finish")

test_size = len(test_cel)
predictions = np.zeros([test_size, 3])

for m_idx, m in enumerate(models):
    print(m_idx, 'Accuarcy:', m.get_accuarcy(test_cel, test_class))
    p = m.softmax(test_cel)
    predictions += p

ensemble_correct_prediction = tf.equal(tf.argmax(predictions, 1), test_class)
ensemble_accuracy = tf.reduce_mean(tf.cast(ensemble_correct_prediction, tf.float32))
print('Ensemble accuarcy : ', sess.run(ensemble_accuracy))