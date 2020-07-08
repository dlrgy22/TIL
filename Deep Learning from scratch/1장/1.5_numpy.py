import numpy as np
def element_wise(x,y):
    print(x + y)
    print(x - y)
    print(x * y)
    print(x / y)

def two_Dimension(A,B):
    print(A)
    #행렬의 형상
    print(A.shape)
    #원소의 자료형
    print(A.dtype)
    print(A + B)

def broadcast(A,C):
    print(A * C)
    print(A + C)
x = np.array([1.0,2.0,3.0])
y = np.array([2.0,4.0,6.0])
element_wise(x,y)
print()
A = np.array([[1,2],[3,4]])
B = np.array([[3,0],[6,0]])
C = np.array([10,20])
two_Dimension(A,B)
print()
broadcast(A,C)
