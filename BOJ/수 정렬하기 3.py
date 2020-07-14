import sys
n = int(sys.stdin.readline())
number = [0 for i in range(10001)]
for i in range(n):
    number[int(sys.stdin.readline())] += 1
for i in range(1, 10001):
    for j in range(number[i]):
        print(i)
