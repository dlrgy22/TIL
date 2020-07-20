import sys
number = list(sys.stdin.readline().replace('\n',''))
length = len(number)
dp = [0 for i in range(length + 1)]

if number[0] == '0':
    print(0)
    sys.exit()

number = [0] + number
dp[0] = 1
dp[1] = 1
for i in range(2, length + 1):
    check_num = int(number[i - 1]) * 10 + int(number[i])
    if number[i] != '0':
        dp[i] += dp[i - 1]
    if check_num <= 26 and check_num >= 10:
        dp[i] += dp[i - 2]
    dp[i] %= 1000000
print(dp[length])