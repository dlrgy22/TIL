import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from keras.utils import to_categorical

def get_data(path):
    df = pd.read_csv(path)
    y_data = df.values[:, 0]
    x_data = df.values[:, 1:]
    return np.array(x_data), np.array(y_data)

def split_val_data(x_data, y_data):
    x_train, x_val, y_train, y_val = train_test_split(x_data, y_data, test_size=0.1, random_state=1234)
    return x_train, x_val, y_train, y_val

def std_scale(x_data):
    standardScaler = StandardScaler()
    standardScaler.fit(x_data)
    scaled_data = standardScaler.transform(x_data)
    return scaled_data

def one_hot(y_data):
    y_data[np.where(y_data == 'PH')] = 0
    y_data[np.where(y_data == 'HI')] = 1
    y_data[np.where(y_data == 'EL')] = 2
    y_data[np.where(y_data == 'CO')] = 3
    y_data[np.where(y_data == 'GR')] = 4
    y_data[np.where(y_data == 'MI')] = 5
    y_one_hot = to_categorical(y_data, num_classes=6)
    y_one_hot.reshape((y_one_hot.shape[0], y_one_hot.shape[1], 1))
    return y_one_hot