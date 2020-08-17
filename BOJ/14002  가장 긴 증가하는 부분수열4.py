import sys
from collections import deque
n = int(sys.stdin.readline())
num_arr = list(map(int, sys.stdin.readline().split()))
dp = [1 for i in range(n)]
idx = [-1 for i in range(n)]
large = 1
index = 0
dp[0] = 1
answer = deque()
for i in range(1, n):
    for j in range(0, i):
        if num_arr[j] < num_arr[i]:
            if dp[j] + 1 > dp[i]:
                dp[i] = dp[j] + 1
                idx[i] = j
    if large < dp[i]:
        large = dp[i]
        index = i
print(large)
while index != -1:
    answer.appendleft(num_arr[index])
    index = idx[index]
for element in answer:
    print(element, end = ' ')