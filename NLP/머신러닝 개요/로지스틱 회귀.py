import torch
import numpy as np
from torch import nn, optim
from torch.nn import functional as F

class BinaryClassifier(nn.Module):
    def __init__(self, input_features):
        super().__init__()
        self.linear = nn.Linear(input_features, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.linear(x)
        return self.sigmoid(x)

if __name__ == "__main__":
    
    def get_network(input_features):
        model = BinaryClassifier(input_features)
        loss_function = nn.BCELoss()
        optimizer = optim.SGD(model.parameters(), lr=0.01)

        return model, loss_function, optimizer

    def train_model(x_train, y_train, epochs):
        print("Start training")

        model.train()
        for epoch in range(epochs):
            y_pred = model(x_train)
            loss = loss_function(y_pred, y_train)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if epoch % 20 == 0:
                prediction = y_pred >= torch.FloatTensor([0.5])
                correct_prediction = prediction.float() == y_train
                acc = correct_prediction.sum().item() / len(correct_prediction) * 100
                print(f"Epoch {epoch}/{epochs} loss : {loss} acc : {acc}")

    torch.manual_seed(1)
    x_data = [[1, 2], [2, 3], [3, 1], [4, 3], [5, 3], [6, 2]]
    y_data = [[0], [0], [0], [1], [1], [1]]
    x_train = torch.FloatTensor(x_data)
    y_train = torch.FloatTensor(y_data)

    model, loss_function, optimizer = get_network(2)
    train_model(x_train, y_train, 10000)


    