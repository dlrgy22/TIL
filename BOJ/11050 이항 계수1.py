import sys
n, k = map(int, sys.stdin.readline().split())
answer = 1
for i in range(n, n - k, -1):
    answer*=i
for i in range(1, k + 1):
    answer //= i
print(answer)