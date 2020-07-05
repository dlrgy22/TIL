import sys
n, k = map(int,sys.stdin.readline().split())
coin = []
dp = [0 for i in range(k + 1)]
dp[0] = 1
for i in range(n):
    coin.append(int(sys.stdin.readline()))
for i in coin:
    for j in range(i, k + 1):
        dp[j] = dp[j] + dp[j - i]
print(dp[k])