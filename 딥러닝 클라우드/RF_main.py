import data_processing
import model
import numpy as np
from sklearn.model_selection import KFold


def cross_validation(x_data, y_data):
    kf = KFold(n_splits=10, random_state = 1234, shuffle = True)
    acc_list = np.zeros(10)
    val_acc_list = np.zeros(10)
    no = 0
    for train_idx, test_idx in kf.split(x_data):
        x_train, x_test = x_data[train_idx], x_data[test_idx]
        y_train, y_test = y_data[train_idx], y_data[test_idx]
        x_train, x_test = data_processing.robuster_scale(x_train, x_test)
        x_train, y_train = data_processing.over_sampling(x_train, y_train)
        Model = model.RF(x_train, y_train, x_test, y_test)
        Model.train()
        acc, test_acc = Model.acc()
        acc_list[no] = acc
        val_acc_list[no] = test_acc
        no += 1
    print(acc_list, val_acc_list)
    return np.mean(acc_list), np.mean(val_acc_list)

path = './trainset.csv'
x_data, y_data, name = data_processing.get_data(path)
print(cross_validation(x_data, y_data))