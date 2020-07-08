import numpy as np
import matplotlib.pyplot as plt

x = np.arange(0,6,0.1)
y1 = np.sin(x)
y2 = np.cos(x)

plt.plot(x,y1,label = "sinx")
plt.plot(x,y2,label = "cosx",color = 'red',linestyle = '--')
plt.legend()
plt.xlabel("x")
plt.ylabel("y")
plt.title("sin & cos")
plt.show()