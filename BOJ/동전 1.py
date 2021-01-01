import sys

n, k = map(int, sys.stdin.readline().split())
coin = []
for i in range(n):
    coin.append(int(sys.stdin.readline()))

dp = [0 for i in range(k + 1)]
dp[0] = 1

for i in range(n):
    for j in range(coin[i], k + 1):
        dp[j] = dp[j - coin[i]] + dp[j]

print(dp[k])