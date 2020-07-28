import sys
n = int(sys.stdin.readline())
num_list = list(map(int, sys.stdin.readline().split()))
bigger_dp = [0 for i in range(n)]
bigger_sv =[0 for i in range(n)]
smaller_dp = [0 for i in range(n)]
smaller_sv = [0 for i in range(n)]

bigger_dp[0] = 1
bigger_sv[0] = [1, num_list[0]]
num_sv = 0
for i in range(1, n):
    large = 0
    for j in range(i):
        if num_list[i] > num_list[j] and bigger_dp[j] > large:
            large = bigger_dp[j]

    bigger_dp[i] = large + 1
    if bigger_sv[i - 1][0] > bigger_dp[i]:
        bigger_sv[i] = bigger_sv[i - 1]
    else:
        bigger_sv[i] = [bigger_dp[i], num_list[i]]


smaller_dp[n - 1] = 1
smaller_sv[n - 1] = [1, smaller_dp[n - 1]]
for i in range(n - 2, -1, -1):
    large = 0
    for j in range(n - 1, i, -1):
        if num_list[i] > num_list[j] and smaller_dp[j] > large:
            large = smaller_dp[j]
    smaller_dp[i] = large + 1
    if smaller_sv[i + 1][0] > smaller_dp[i]:
        smaller_sv[i] = smaller_sv[i + 1]
    else:
        smaller_sv[i] = [smaller_dp[i], num_list[i]]

result = smaller_sv[0][0]
for i in range(n - 1):
    if bigger_sv[i][1] == smaller_sv[i + 1][1]:
        result = max(bigger_sv[i][0] + smaller_sv[i + 1][0] - 1, result)
    else:
        result = max(bigger_sv[i][0] + smaller_sv[i + 1][0], result)


result = max(bigger_sv[n - 1][0], result)
print(result)