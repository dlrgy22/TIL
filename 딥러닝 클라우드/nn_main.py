import data_processing
import model

path = './trainset.csv'
x_data, y_data = data_processing.get_data(path)
x_data = data_processing.std_scale(x_data)
y_data = data_processing.one_hot(y_data)
x_train, x_val, y_train, y_val = data_processing.split_val_data(x_data, y_data)

Model = model.NN_model(x_train, y_train, x_val, y_val)
Model.train()