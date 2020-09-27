import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Flatten
from keras.utils import to_categorical
from keras import optimizers
from matplotlib import pyplot as plt


def get_test_data():
    df = pd.read_csv('train.csv')
    idx = df[df['i'] < 0].index
    df = df.drop(idx)
    Classification_data = df['class']
    df = df.drop('class', axis=1)

    df['d_dered_u'] = df['dered_u'] - df['u']
    df['d_dered_g'] = df['dered_g'] - df['g']
    df['d_dered_r'] = df['dered_r'] - df['r']
    df['d_dered_i'] = df['dered_i'] - df['i']
    df['d_dered_z'] = df['dered_z'] - df['z']

    df['d_dered_rg'] = df['dered_r'] - df['dered_g']
    df['d_dered_ig'] = df['dered_i'] - df['dered_g']
    df['d_dered_zg'] = df['dered_z'] - df['dered_g']
    df['d_dered_ri'] = df['dered_r'] - df['dered_i']
    df['d_dered_rz'] = df['dered_r'] - df['dered_z']
    df['d_dered_iz'] = df['dered_i'] - df['dered_z']
    df.drop(
        ['airmass_z', 'airmass_i', 'airmass_r', 'airmass_g','nDetect'],
        axis=1, inplace=True)

    celestial_data = df.values[:, 1:-1]


    return celestial_data, Classification_data

x_train, y_train = get_test_data()
mean = x_train.mean(axis=0)
std = x_train.std(axis=0)
x_train = (x_train - mean) / std
x_train = x_train.reshape((x_train.shape[0], x_train.shape[1], 1))
y_train = to_categorical(y_train, num_classes=3)

model = Sequential()
model.add(Flatten())
model.add(Dense(88, activation='relu', kernel_initializer='he_normal'))
model.add(Dense(3, activation='softmax', kernel_initializer='he_normal'))
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
adam = optimizers.Adam(lr=0.3)
hist = model.fit(x_train, y_train, batch_size=64, epochs=50, verbose=1)

plt.plot(hist.history['loss'])
plt.show()
plt.plot(hist.history['acc'])
plt.show()