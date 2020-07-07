import sys
n = int(sys.stdin.readline())
cost = [0]
cost += map(int,sys.stdin.readline().split())
dp = [0 for i in range(n+1)]
dp[0] = 0
for i in range(1,n + 1):
    for j in range(0,i + 1):
        dp[i] = max(dp[i - j] + cost[j],dp[i])
print(dp[n])