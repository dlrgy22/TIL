import sys
n, k = map(int,sys.stdin.readline().split())
dp = [[1 for i in range(n + 1)] for j in range(k + 1)]
for i in range(2,k + 1):
    for j in range(n + 1):
        dp[i][j] = sum(dp[i - 1][:j + 1]) % 1000000000

print(dp[k][n])