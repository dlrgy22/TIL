import data_processing
import model
import numpy as np
from sklearn.model_selection import KFold

def cross_validation(x_data, y_data):
    kf = KFold(n_splits=10, random_state = 1234, shuffle = True)
    acc = np.zeros(10)
    val_acc = np.zeros(10)
    no = 0
    for train_idx, test_idx in kf.split(x_data):
        x_train, x_test = x_data[train_idx], x_data[test_idx]
        y_train, y_test = y_data[train_idx], y_data[test_idx]
        x_train = x_train.reshape(3852, 31, -1)
        x_test = x_test.reshape(428, 31, -1)
        Model = model.CNN_model(x_train, y_train, x_test, y_test)
        Model.train()
        acc = Model.hist.history['accuracy'][-1]
        val_acc[no] = Model.hist.history['val_accuracy'][-1]
        no += 1

    return np.mean(acc), np.mean(val_acc)

path = './trainset.csv'
x_data, y_data = data_processing.get_data(path)
x_data = data_processing.std_scale(x_data)
y_data = data_processing.one_hot(y_data)
print(cross_validation(x_data, y_data))