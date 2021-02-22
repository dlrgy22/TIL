import sys
T = int(sys.stdin.readline())
dp = [0 for i in range(100 + 1)]
dp[:5] = [0, 1, 1, 1, 2]
for i in range(5, 101):
    dp[i] = dp[i - 1] + dp[i -5]
for i in range(T):
    n = int(sys.stdin.readline())
    print(dp[n])