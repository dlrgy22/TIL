def solution(triangle):
    dp = [[] for i in range(len(triangle))]
    dp[0].append(triangle[0][0])
    for i in range(1, len(triangle)):
        for j in range(len(triangle[i])):
            if j == 0:
                dp[i].append(dp[i - 1][0] + triangle[i][0])
            elif j == len(triangle[i]) - 1:
                dp[i].append(dp[i - 1][j - 1] + triangle[i][j])
            else:
                dp[i].append(max(dp[i - 1][j - 1] + triangle[i][j], dp[i - 1][j] + triangle[i][j]))

    answer = max(dp[len(triangle)- 1])
    return answer

triangle = [[7], [3, 8], [8, 1, 0], [2, 7, 4, 4], [4, 5, 2, 6, 5]]
print(solution(triangle))