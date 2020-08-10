import sys
n = int(sys.stdin.readline())
num_list = list(map(int, sys.stdin.readline().split()))
dp = [num_list[0]]
length = 1
for i in range(1, n):
    if dp[length - 1] < num_list[i]:
        dp.append(num_list[i])
        length += 1
    else:
        for idx in range(len(dp)):
            if dp[idx] >= num_list[i]:
                dp[idx] = num_list[i]
                break
print(length)
