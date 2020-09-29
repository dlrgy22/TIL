import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Flatten
from keras.utils import to_categorical
from keras import optimizers
from keras.layers import Dropout
from keras.layers.convolutional import Conv1D
from keras.layers.convolutional import MaxPooling1D
from matplotlib import pyplot as plt
from keras.layers import BatchNormalization

def get_train_data():
    df = pd.read_csv('./train.csv')
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
    celestial_data = df.values[:, 1:]
    return celestial_data, Classification_data

def get_test_data():
    df = pd.read_csv('./test.csv')
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
        ['airmass_z', 'airmass_i', 'airmass_r', 'airmass_g', 'nDetect'],
        axis=1, inplace=True)
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

x, y = get_train_data()
mean = x.mean(axis=0)
std = x.std(axis=0)
x = (x - mean) / std
x_train = x[1000:]
y_train = y[1000:]

x_test = x[:1000]
y_test = y[:1000]

x_train = x_train.reshape((x_train.shape[0], x_train.shape[1], 1))
y_train = to_categorical(y_train, num_classes=3)

x_test = x_test.reshape((x_test.shape[0], x_test.shape[1], 1))
y_test = to_categorical(y_test, num_classes=3)

model = Sequential()
model.add(Flatten())
model.add(Dense(138, activation='relu', kernel_initializer='he_normal'))
model.add(Dense(46, activation='relu', kernel_initializer='he_normal'))
model.add(Dense(3, activation='softmax', kernel_initializer='he_normal'))
adam = optimizers.Adam(lr=0.1)
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
hist = model.fit(x_train, y_train, validation_data=(x_test, y_test), epochs=300, verbose=2, class_weight={0:0., 1:1.5, 2:1.5})

plt.plot(hist.history['loss'])
plt.show()
plt.plot(hist.history['accuracy'])
plt.show()

pred = model.predict(x_train)
length = len(pred)
result = []
answer = []
for i in range(length):
    result.append(np.argmax(pred[i]))
    answer.append(np.argmax(y_train[i]))

zero = 0
one = 0
two = 0
zero_r = 0
one_r = 0
two_r = 0

for i in range(length):
    if result[i] != answer[i]:
        if answer[i] == 0:
            zero += 1
        elif answer[i] == 1:
            one += 1
        else:
            two += 1

        if result[i] == 0:
            zero_r += 1
        elif result[i] == 1:
            one_r += 1
        else:
            two_r += 1

print(zero, one, two)
print(zero_r, one_r, two_r)

test_data = get_test_data()
test_data = (test_data - mean) / std
test_data = test_data.reshape((test_data.shape[0], test_data.shape[1], 1))
pred = model.predict(test_data, verbose = 1)
length = len(pred)
result = []
for i in range(length):
    result.append(np.argmax(pred[i]))

make_submission(result)