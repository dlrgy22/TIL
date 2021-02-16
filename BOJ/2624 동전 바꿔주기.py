import sys

T = int(sys.stdin.readline())
K = int(sys.stdin.readline())
coin = []

for i in range(K):
    p, n = map(int, sys.stdin.readline().split())
    coin.append([p, n])

coin.sort()
dp = [[0 for i in range(T + 1)] for j in range(101)]
for i in range(K + 1):
    dp[i][0] = 1

for i in range(1, K + 1):
    for j in range(1, coin[i - 1][1] + 1):
        for k in range(0, T + 1):
            idx = k - coin[i - 1][0] * j
            if j == 1:
                dp[i][k] = dp[i - 1][k]

            if idx >= 0:
                dp[i][k] += dp[i - 1][idx]

print(dp[K][T])