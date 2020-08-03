import numpy as np

class Adagrad:
    def __init__(self, lr = 0.01):
        self.lr = lr
        self.h = None

    def update(self, params, grads):
        if self.h is None:
            self.h = {}
            for key, val in params.items():
                self.h[key] = np.zeros_like(val)

        for key in params.key():
            self.h[key] = grads[key] * grads[key]
            params[key] -= self.lr * grads[key] / (np.sqrt(self.h) + 1e-7)