import sys
n = int(sys.stdin.readline())
number = list(map(int,sys.stdin.readline().split()))
dp = [1]*n
result = 0
for i in range(n):
    large =1
    for j in range(i):
        if number[i] < number[j]:
            large = max(large,dp[j]+1)
    dp[i] = large
    result = max(large,result)
print(result)