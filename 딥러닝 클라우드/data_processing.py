import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from keras.utils import to_categorical
from imblearn.over_sampling import SMOTE

def get_data(path):
    df = pd.read_csv(path)
    y_data = df.values[:, 0]
    x_data = df.values[:, 1:]
    return np.array(x_data), np.array(y_data)

def split_val_data(x_data, y_data):
    x_train, x_val, y_train, y_val = train_test_split(x_data, y_data, test_size=0.2, random_state=1234)
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
    print(len(np.where(y_data == 0)[0]))
    print(len(np.where(y_data == 1)[0]))
    print(len(np.where(y_data == 2)[0]))
    print(len(np.where(y_data == 3)[0]))
    print(len(np.where(y_data == 4)[0]))
    print(len(np.where(y_data == 5)[0]))
    y_one_hot = to_categorical(y_data, num_classes=6)
    y_one_hot.reshape((y_one_hot.shape[0], y_one_hot.shape[1], 1))
    return y_one_hot

def over_sampling(x_train, y_train):
    smote = SMOTE(random_state=1234)
    x_train, y_train = smote.fit_sample(x_train, y_train)
    return x_train, y_train