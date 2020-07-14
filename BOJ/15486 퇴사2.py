import sys
n = int(sys.stdin.readline())
counseling = [[0, 0]]
for i in range(n):
    counseling.append(list(map(int,sys.stdin.readline().split())))
dp = [0 for i in range(n + 1)]
for i in range(1, n + 1):
    if(i + counseling[i][0] - 1 <= n):
        dp[i + counseling[i][0] - 1] = max(dp[i - 1] + counseling[i][1], dp[i + counseling[i][0] - 1])
    dp[i] = max(dp[i - 1], dp[i])
print(dp[n])