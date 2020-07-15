import sys
import math
def return_per(length,number):
    per = list(range(1,length + 1))
    while number != 0 and length != 0:
        div = math.factorial(length - 1)
        idx = number // div
        if number % div == 0:
            idx -= 1
        print(per[idx], end = ' ')
        del per[idx]
        number = number % div
        if number == 0:
            number = div
        length -= 1

    for i in per:
        print(i, end = ' ')

def return_num(length, per):
    count = 1
    number = list(range(1, length + 1))
    for i in per:

        idx = number.index(i)
        count += (idx * math.factorial(length - 1))
        del number[idx]
        length -= 1
    print(count)


n = int(sys.stdin.readline())
Q = list(map(int,sys.stdin.readline().split()))
if Q[0] == 1:
    return_per(n, Q[1])
elif Q[0] == 2:
    return_num(n, tuple(Q[1:]))
