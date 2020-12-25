def solution(n, money):
    dp = [[0 for i in range(n + 1)] for j in range(len(money))]
    money = sorted(money)

    for i in range(len(money)):
        dp[i][0] = 1

    for i in range(len(money)):
        for j in range(n + 1):
            idx = j - money[i]
            if idx >= 0:
                if i == 0:
                    dp[i][j] = dp[i][idx]
                else:
                    dp[i][j] = dp[i][idx] + dp[i - 1][j]

            elif i != 0:
                dp[i][j] = dp[i - 1][j]

    return dp[len(money) - 1][n]

n = 5
money = [1, 2, 5]
print(solution(n, money))