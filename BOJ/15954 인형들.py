import sys
import math

def mean(sum, n):
    return sum/n

def std(num, mean, n):
    std = 0
    for i in range(n):
        std += (num[i] - mean) ** 2
    return (std/n) ** 0.5

n, k = map(int, sys.stdin.readline().split())
num_prefer = list(map(int, sys.stdin.readline().split()))
result = math.inf

while k <= n:
    for i in range(n - k + 1):
        m = mean(sum(num_prefer[i : i + k]), k)
        s = std(num_prefer[i : i + k], m, k)
        result = min(s, result)
    k += 1

print('%.7f' %(result))