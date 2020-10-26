from matplotlib import pyplot as plt
from keras.models import Sequential
from keras.layers import Dense
from keras import optimizers
from keras.layers import Flatten
from keras.layers import Dropout
from keras.layers import BatchNormalization
from keras.layers.convolutional import Conv1D
from keras.layers.convolutional import MaxPooling1D
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics
import xgboost
from sklearn import svm

class NN_model:
    def __init__(self, x_train, y_train, x_val=[], y_val=[]):
        self.x_train = x_train
        self.y_train = y_train
        self.x_val = x_val
        self.y_val = y_val


    def train(self):
        self.model = Sequential()

        self.model.add(Dense(64, activation='relu', kernel_initializer='he_normal'))
        self.model.add(BatchNormalization())
        self.model.add(Dropout(0.3))
        self.model.add(Dense(16, activation='relu', kernel_initializer='he_normal'))
        self.model.add(BatchNormalization())
        self.model.add(Dropout(0.3))

        self.model.add(Dense(6, activation='softmax', kernel_initializer='he_normal'))
        adam = optimizers.adam(lr=0.001)
        self.model.compile(optimizer=adam, loss='categorical_crossentropy', metrics=['accuracy'])
        if len(self.x_val) != 0:
            self.hist = self.model.fit(self.x_train, self.y_train, validation_data=(self.x_val, self.y_val), batch_size=32, epochs=1000, verbose=2)
        else:
            self.hist = self.model.fit(self.x_train, self.y_train, batch_size=32, epochs=1000, verbose=2)


    def plot_hist(self):

        plt.plot(self.hist.history['loss'])
        plt.plot(self.hist.history['val_loss'])
        plt.show()

        plt.plot(self.hist.history['acc'])
        plt.plot(self.hist.history['val_acc'])
        plt.show()

    def predict(self, test_data):
        self.pred = self.model.predict(test_data)
        self.length = len(self.pred)
        self.result = []
        for i in range(self.length):
            self.result.append(np.argmax(self.pred[i]))
        return self.result



class CNN_model:
    def __init__(self, x_train, y_train, x_val = [], y_val = []):
        self.x_train = x_train
        self.y_train = y_train
        self.x_val = x_val
        self.y_val = y_val


    def train(self):
        self.model = Sequential()
        self.model.add(Conv1D(filters=16, kernel_size=2, activation='relu', kernel_initializer='he_uniform', input_shape=(17,1)))

        self.model.add(MaxPooling1D(pool_size=2, padding='same'))
        self.model.add(Dropout(0.25))

        self.model.add(Conv1D(filters=32, kernel_size=2, activation='relu', kernel_initializer='he_uniform'))
        self.model.add(BatchNormalization())
        self.model.add(MaxPooling1D(pool_size=2, padding='same'))
        self.model.add(Dropout(0.25))

        self.model.add(Flatten())
        self.model.add(Dense(64, activation='relu', kernel_initializer='he_uniform'))
        self.model.add(BatchNormalization())
        self.model.add(Dropout(0.3))
        self.model.add(Dense(6, activation='softmax', kernel_initializer='he_uniform'))
        adam = optimizers.Adam(lr=0.003)
        self.model.compile(optimizer=adam, loss='categorical_crossentropy', metrics=['accuracy'])
        if len(self.x_val) != 0:
            self.hist = self.model.fit(self.x_train, self.y_train, epochs=300, validation_data=(self.x_val, self.y_val), verbose=2, batch_size=32)
        else:
            self.hist = self.model.fit(self.x_train, self.y_train, epochs=1000, verbose=2, batch_size=32)

    def plot_hist(self):
        print(self.hist.history)

        plt.plot(self.hist.history['loss'])
        plt.plot(self.hist.history['val_loss'])
        plt.show()

        # plt.plot(self.hist.history['accuracy'])
        # plt.plot(self.hist.history['val_accuracy'])
        plt.plot(self.hist.history['acc'])
        plt.plot(self.hist.history['val_acc'])
        plt.show()

    def predict(self, test_data):
        self.pred = self.model.predict(test_data)
        self.length = len(self.pred)
        self.result = []
        for i in range(self.length):
            self.result.append(np.argmax(self.pred[i]))
        return self.result


class RF:
    def __init__(self, x_train, y_train, x_val=[], y_val=[]):
        self.x_train = x_train
        self.y_train = y_train
        self.x_val = x_val
        self.y_val = y_val

    def train(self):
        self.model = RandomForestClassifier(n_estimators=200,
                                            random_state=1234,
                                            max_features='sqrt',
                                            min_samples_split=2,
                                            min_samples_leaf=1,
                                            bootstrap=False,
                                            )
        self.model.fit(self.x_train, self.y_train)

    def acc(self):
        self.pred_train = self.model.predict(self.x_train)
        self.train_acc = metrics.accuracy_score(self.pred_train, self.y_train)
        if len(self.x_val) != 0:
            self.pred_test = self.model.predict(self.x_val)
            self.test_acc = metrics.accuracy_score(self.pred_test, self.y_val)
            return self.train_acc, self.test_acc

        return self.train_acc

    def predict(self, test_data):
        pred = self.model.predict(test_data)
        return pred



class XGB:
    def __init__(self, x_train, y_train, x_val = [], y_val = []):
        self.x_train = x_train
        self.y_train = y_train
        self.x_val = x_val
        self.y_val = y_val

    def train(self):
        self.model = xgboost.XGBClassifier(n_estimators=400,
                                           n_jobs=-1,
                                           learning_rate=0.1,
                                           random_state=1234,
                                           )
        self.model.fit(self.x_train, self.y_train)

    def acc(self):
        self.pred_train = self.model.predict(self.x_train)
        self.train_acc = metrics.accuracy_score(self.pred_train, self.y_train)
        if len(self.x_val) != 0:
            self.pred_test = self.model.predict(self.x_val)
            self.test_acc = metrics.accuracy_score(self.pred_test, self.y_val)
            return self.train_acc, self.test_acc

        return self.train_acc

    def predict(self, test_data):
        pred = self.model.predict(test_data)
        return pred


class SVM:
    def __init__(self, x_train, y_train, x_val = [], y_val = []):
        self.x_train = x_train
        self.y_train = y_train
        self.x_val = x_val
        self.y_val = y_val

    def train(self,C_val, gamma_val):
        self.model = svm.SVC(kernel='linear', C=C_val, gamma=gamma_val, random_state=1234)
        self.model.fit(self.x_train, self.y_train)

    def acc(self):
        self.pred_train = self.model.predict(self.x_train)
        self.train_acc = metrics.accuracy_score(self.pred_train, self.y_train)
        if len(self.x_val) != 0:
            self.pred_test = self.model.predict(self.x_val)
            self.test_acc = metrics.accuracy_score(self.pred_test, self.y_val)
            return self.train_acc, self.test_acc

        return self.train_acc

    def predict(self, test_data):
        pred = self.model.predict(test_data)
        return pred

