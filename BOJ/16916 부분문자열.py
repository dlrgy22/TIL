import sys

string1 = sys.stdin.readline().replace('\n', '')
string2 = sys.stdin.readline().replace('\n', '')

dp = [[0 for i in range(len(string2))] for j in range(len(string1))]
for i in range(len(string1)):
    for j in range(len(string2)):
        if i == 0:
            if string2[j] == string1[i]:
                dp[i][j] = 1
            elif j != 0:
                dp[i][j] = dp[i][j - 1]
            else:
                dp[i][j] = 0
        else:
            if string2[j] == string1[i]:
                if j != 0:
                    dp[i][j] = max(dp[i - 1][j - 1] + 1, dp[i][j - 1], dp[i - 1][j])
                else:
                    dp[i][j] = 1
            else:
                if j == 0:
                    dp[i][j] = dp[i - 1][j]
                else:
                    dp[i][j] = max(dp[i][j - 1], dp[i - 1][j])


print(dp[len(string1) - 1][len(string2) - 1])