import numpy as np
import matplotlib.pyplot as plt


def U(x, r, k):
    return (k*(x-r)**2)/2

def F(x):
    return U(x, 10000, 10)+U(x, 90000, 20)

x=[]
y=[]
for i in range(100000):
    x.append(i)
    y.append(F(i))

plt.plot(x, y)
plt.show()