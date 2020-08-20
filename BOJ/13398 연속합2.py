import sys

n = int(sys.stdin.readline())
num_list = list(map(int, sys.stdin.readline().split()))
minus_idx = []
sum = 0
dp = [[0,0] for i in range(n)]
dp[0][0] = num_list[0]
answer = num_list[0]
for i in range(1, n):
    dp[i][0] = max(dp[i - 1][0] + num_list[i], num_list[i])
    dp[i][1] = max(dp[i - 1][0], dp[i - 1][1] + num_list[i])
    answer = max(dp[i][0], dp[i][1], answer)

print(answer)