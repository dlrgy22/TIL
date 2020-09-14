import numpy as np
import pandas as pd
from sklearn import metrics
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers import Dropout
from keras.layers import BatchNormalization
from keras.layers.convolutional import Conv1D
from keras.layers.convolutional import MaxPooling1D
from keras.utils import to_categorical
from keras import optimizers



def make_submission(classification):
    head = ['id', 'class']
    data = []
    for i in range(len(classification)):
        data.append([320000 + i, classification[i]])
    data = np.vstack((head, data))
    file_name = "submission.csv"
    np.savetxt(file_name, data, fmt="%s", delimiter=",")

def get_test_data():
    df = pd.read_csv('test.csv')
    celestial_data = df.values[:, 1:]
    return celestial_data

def get_train_data():
    df = pd.read_csv('train.csv')
    celestial_data = df.values[:, 1:-1]
    Classification_data = df.values[:, -1]

    return celestial_data, Classification_data

x_train, y_train = get_train_data()
mean = x_train.mean(axis = 0)
std = x_train.std(axis = 0)
x_train = (x_train - mean) / std
y_train = to_categorical(y_train, num_classes=3)

x_test = x_train[:1000]
y_test = y_train[:1000]
x_train = x_train[1000:]
y_train = y_train[1000:]
x_train = x_train.reshape((x_train.shape[0], x_train.shape[1], 1))
x_test = x_test.reshape((x_test.shape[0], x_test.shape[1], 1))

model = Sequential()
model.add(Conv1D(filters=32, kernel_size = 3, activation = 'relu',input_shape=(18, 1)))
model.add(MaxPooling1D(pool_size=2, padding = 'same'))
model.add(BatchNormalization())
model.add(Conv1D(filters=64, kernel_size = 3, activation = 'relu',input_shape=(18, 1)))
model.add(MaxPooling1D(pool_size=2, padding = 'same'))
model.add(BatchNormalization())
model.add(Flatten())
model.add(Dropout(0.5))
model.add(Dense(36, activation='relu', kernel_initializer='he_normal'))
model.add(Dropout(0.25))
model.add(Dense(18, activation='relu', kernel_initializer='he_normal'))
model.add(Dense(3, activation='softmax', kernel_initializer='he_normal'))
model.compile(optimizer='adam', loss = 'categorical_crossentropy', metrics=['accuracy'])
adam = optimizers.Adam(lr = 0.01)
hist = model.fit(x_train, y_train, validation_data=(x_test, y_test), epochs=10, verbose=1)

test_data = get_test_data()
test_data = (test_data - mean) / std
test_data = test_data.reshape((test_data.shape[0], test_data.shape[1], 1))

pred = model.predict(test_data, verbose = 1)
length = len(pred)
result = []
for i in range(length):
    result.append(np.argmax(pred[i]))

make_submission(result)
print(result)