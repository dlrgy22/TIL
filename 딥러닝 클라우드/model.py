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
from sklearn.model_selection import  KFold

class NN_model:
    def __init__(self, x_train, y_train, x_val, y_val):
        self.x_train = x_train
        self.y_train = y_train
        self.x_val = x_val
        self.y_val = y_val


    def train(self):
        self.model = Sequential()
        # self.model.add(Dense(1024, activation='relu', kernel_initializer='he_normal'))
        # self.model.add(Dropout(0.5))
        self.model.add(Dense(512, activation='relu', kernel_initializer='he_normal'))
        self.model.add(Dropout(0.5))
        self.model.add(Dense(256, activation='relu', kernel_initializer='he_normal'))
        self.model.add(Dropout(0.5))
        self.model.add(Dense(6, activation='softmax', kernel_initializer='he_normal'))
        adam = optimizers.Adam(lr=0.001 )
        self.model.compile(optimizer=adam, loss='categorical_crossentropy', metrics=['accuracy'])
        self.hist = self.model.fit(self.x_train, self.y_train, validation_data=(self.x_val, self.y_val), batch_size=64, epochs=100, verbose=2)

    def plot_hist(self):
        print(self.hist.history)

        plt.plot(self.hist.history['loss'])
        plt.plot(self.hist.history['val_loss'])
        plt.show()

        plt.plot(self.hist.history['accuracy'])
        plt.plot(self.hist.history['val_accuracy'])
        plt.show()

    def predict(self):
        self.pred = self.model.predict(self.test_data)
        self.length = len(self.pred)
        self.result = []
        for i in range(self.length):
            self.result.append(np.argmax(self.pred[i]))
        return self.result



class CNN_model:
    def __init__(self, x_train, y_train,x_val, y_val):
        self.x_train = x_train
        self.y_train = y_train
        self.x_val = x_val
        self.y_val = y_val


    def train(self):
        self.model = Sequential()
        self.model.add(Conv1D(filters=32, kernel_size=3, activation='relu', kernel_initializer='he_uniform', input_shape=(31, 1)))
        self.model.add(MaxPooling1D(pool_size=2, padding='same'))
        self.model.add(BatchNormalization())
        self.model.add(Dropout(0.2))

        self.model.add(Conv1D(filters=32, kernel_size=3, activation='relu', kernel_initializer='he_uniform'))
        self.model.add(MaxPooling1D(pool_size=2, padding='same'))
        self.model.add(BatchNormalization())
        self.model.add(Dropout(0.2))

        self.model.add(Conv1D(filters=64, kernel_size=3, activation='relu', kernel_initializer='he_uniform'))
        self.model.add(MaxPooling1D(pool_size=2, padding='same'))
        self.model.add(BatchNormalization())
        self.model.add(Dropout(0.2))

        self.model.add(Flatten())
        self.model.add(Dense(138, activation='relu', kernel_initializer='he_uniform'))
        self.model.add(Dropout(0.3))
        self.model.add(Dense(46, activation='relu', kernel_initializer='he_uniform'))
        #self.model.add(Dropout(0.1))
        self.model.add(Dense(6, activation='softmax', kernel_initializer='he_uniform'))
        adam = optimizers.Adam(lr=0.001)
        self.model.compile(optimizer=adam, loss='categorical_crossentropy', metrics=['accuracy'])
        self.hist = self.model.fit(self.x_train, self.y_train, validation_data=(self.x_val, self.y_val), epochs=100, verbose=2)

    def plot_hist(self):
        print(self.hist.history)

        plt.plot(self.hist.history['loss'])
        plt.plot(self.hist.history['val_loss'])
        plt.show()

        plt.plot(self.hist.history['accuracy'])
        plt.plot(self.hist.history['val_accuracy'])
        plt.show()
