import sys
import math
x, y = map(int, sys.stdin.readline().split())
num = y - x
sum = 0
result = 0
while num > sum:
    sum += math.ceil((result + 1) / 2)
    result += 1
print(result)