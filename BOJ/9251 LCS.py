import sys
string1 = sys.stdin.readline().replace('\n', '')
string2 = sys.stdin.readline().replace('\n', '')
n1 = len(string1)
n2 = len(string2)
dp = [[0 for i in range(n2 + 1)] for j in range(n1 + 1)]
for i in range(1, n1 + 1):
    for j in range(1, n2 + 1):
        if string1[i - 1] == string2[j - 1]:
            dp[i][j] = dp[i - 1][j - 1] + 1
        else:
            dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])

print(dp[n1][n2])
