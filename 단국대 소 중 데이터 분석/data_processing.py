import pandas as pd
import numpy as np
from keras.utils import to_categorical

def get_data(path):
    df = pd.read_csv(path)
    if path == 'train.csv':
        idx = df[df['i'] < 0].index
        df = df.drop(idx)
        y = df['class']
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

    x = df.values[:, 1:]
    if path == 'train.csv':
        return x, y
    else:
        return x

def make_submission(classification):
    head = ['id', 'class']
    data = []
    for i in range(len(classification)):
        data.append([320000 + i, classification[i]])
    data = np.vstack((head, data))
    file_name = "submission.csv"
    np.savetxt(file_name, data, fmt="%s", delimiter=",")

def std_scale(x_train, mean, std):
    x_train = (x_train - mean) / std
    x_train = x_train.reshape((x_train.shape[0], x_train.shape[1], 1))

    return x_train

def one_hot(y_train):
    y_one_hot = to_categorical(y_train, num_classes=3)
    y_one_hot.reshape((y_one_hot.shape[0], y_one_hot.shape[1], 1))
    return y_one_hot
