import data_processing
import model
import numpy as np
from sklearn.model_selection import KFold

# def find_long(pred, y_test):
#     CL = []
#     for i in range(len(y_test)):
#         CL.append(np.argmax(y_test[i]))
#     arr1 = [0 for i in range(6)]
#     arr2 = [0 for i in range(6)]
#     total = [0 for i in range(6)]
#     for i in range(len(pred)):
#
#         if pred[i] != CL[i]:
#             arr1[pred[i]] += 1
#             arr2[CL[i]] += 1
#
#     print(arr1)
#     print(arr2)
#     print(total)


def cross_validation(x_data, y_data):
    kf = KFold(n_splits=5, random_state = 1234, shuffle = True)
    acc = np.zeros(5)
    val_acc = np.zeros(5)
    no = 0
    for train_idx, test_idx in kf.split(x_data):
        x_train, x_test = x_data[train_idx], x_data[test_idx]
        y_train, y_test = y_data[train_idx], y_data[test_idx]
        print(len(x_train), len(y_train))
        x_train, y_train = data_processing.over_sampling(x_train, y_train)
        print(len(x_train), len(y_train))
        Model = model.NN_model(x_train, y_train, x_test, y_test)
        Model.train()
        # acc = Model.hist.history['accuracy'][-1]
        # val_acc[no] = Model.hist.history['val_accuracy'][-1]
        acc = Model.hist.history['acc'][-1]
        val_acc[no] = Model.hist.history['val_acc'][-1]

        no += 1

    return np.mean(acc), np.mean(val_acc)


path = './trainset.csv'
x_data, y_data, f_name = data_processing.get_data(path)
x_data = data_processing.pca(x_data, 10)
x_data = data_processing.std_scale(x_data)
y_data = data_processing.one_hot(y_data)
print(cross_validation(x_data, y_data))



# path = './trainset.csv'
# x_data, y_data, f_name = data_processing.get_data(path)
# x_data, y_data = data_processing.over_sampling(x_data, y_data)
# path  = './testset.csv'
# x_test = data_processing.get_data(path)
# #x_data, x_test = data_processing.std_scale(x_data, x_test)
#
# y_data = data_processing.one_hot(y_data)
# #x_data, x_test = data_processing.pca(x_data, 17, x_test)
# Model = model.NN_model(x_data, y_data)
# Model.train()
# pred = Model.predict(x_test)
# data_processing.save_submission(pred)
#
# #print(cross_validation(x_data, y_data))