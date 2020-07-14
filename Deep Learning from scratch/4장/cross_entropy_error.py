import numpy as np
def cross_entropy_error(y, t):
    delta = 1e-7
    return -np.sum(t * np.log(y + delta))

y1 = [0.1, 0.05, 0.6, 0.0, 0.05, 0.1, 0.0, 0.1, 0.0, 0.0]
y1 = np.array(y1)

y2 = [0.1, 0.05, 0.1, 0.0, 0.05, 0.1, 0.0, 0.6, 0.0, 0.0]
y2 = np.array(y2)
 
t = [0,0,1,0,0,0,0,0,0,0]

print(cross_entropy_error(y1, t))
print(cross_entropy_error(y2, t))
