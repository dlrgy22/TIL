import sys
import math

n = int(sys.stdin.readline())
num_list = list(map(int, sys.stdin.readline().split()))
sum = 0
answer = -math.inf

for i in range(n):
    sum += num_list[i]
    answer = max(sum, answer)
    if sum < 0:
        sum = 0
print(answer)