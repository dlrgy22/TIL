import model
import data_processing

x_train, y_train = data_processing.get_data('train.csv')
mean = x_train.mean(axis=0)
std = x_train.std(axis=0)

x_train = data_processing.std_scale(x_train, mean, std)
y_train = data_processing.one_hot(y_train)
x_train, y_train, x_val, y_val = data_processing.split_val_data(x_train, y_train)

test_data = data_processing.get_data('test.csv')
test_data = data_processing.std_scale(test_data, mean, std)

Model = model.NN_model(x_train, y_train, x_val, y_val, test_data)
Model.train()
Model.plot_hist()
result = Model.predict()
data_processing.make_submission(result)