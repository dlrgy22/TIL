import numpy as np
def sum_squares_error(y, t):
    return 0.5 * np.sum((y - t) ** 2)

y1 = [0.1, 0.05, 0.6, 0.0, 0.05, 0.1, 0.0, 0.1, 0.0, 0.0]
y1 = np.array(y1)

y2 = [0.1, 0.05, 0.1, 0.0, 0.05, 0.1, 0.0, 0.6, 0.0, 0.0]
y2 = np.array(y2)

t = [0,0,1,0,0,0,0,0,0,0]

print(sum_squares_error(y1, t))
print(sum_squares_error(y2, t))

