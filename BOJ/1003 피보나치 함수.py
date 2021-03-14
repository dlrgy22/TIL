import sys

def make_dp():
    dp = [[0, 0] for i in range(41)]
    dp[0] = [1, 0]
    dp[1] = [0, 1]
    for i in range(2, 41):
        for j in range(2):
            dp[i][j] = dp[i - 1][j] + dp[i - 2][j]
    return dp

t = int(sys.stdin.readline())
dp = make_dp()
for i in range(t):
    n = int(sys.stdin.readline())
    print(dp[n][0], dp[n][1])
    