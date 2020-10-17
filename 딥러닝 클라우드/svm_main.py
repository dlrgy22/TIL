import model
import data_processing
import numpy as np
from sklearn.model_selection import KFold


def cross_validation(x_data, y_data, C_val, gamma_val):
    kf = KFold(n_splits=10, random_state = 1234, shuffle = True)
    acc_list = np.zeros(10)
    val_acc_list = np.zeros(10)
    no = 0
    for train_idx, test_idx in kf.split(x_data):
        x_train, x_test = x_data[train_idx], x_data[test_idx]
        y_train, y_test = y_data[train_idx], y_data[test_idx]
        Model = model.SVM(x_train, y_train, x_test, y_test)
        Model.train(C_val, gamma_val)
        acc, test_acc = Model.acc()
        acc_list[no] = acc
        val_acc_list[no] = test_acc
        no += 1
    #print(acc_list)
    #print(val_acc_list)
    return np.mean(acc_list), np.mean(val_acc_list)

path = './trainset.csv'
x_data, y_data, name= data_processing.get_data(path)
x_data = data_processing.std_scale(x_data)

high_avg = 0

for i in range(2,30):
    train_x = data_processing.pca(x_data, i)
    C_list = np.arange(0.1, 1, 0.1)
    gamma_list = np.arange(0.1, 1, 0.1)

    for C_val in C_list:
        for gamma_val in gamma_list:
            acc, val_acc = cross_validation(train_x, y_data, C_val, gamma_val)
            print(acc, val_acc)
            if val_acc > high_avg:
                high_avg = val_acc
                parm = [i, C_val, gamma_val]
print(high_avg)
print(parm)

