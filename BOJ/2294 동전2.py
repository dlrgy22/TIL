import sys
n,k = map(int,sys.stdin.readline().split())
dp=[10001 for i in range(k+1)]
coin =[]
dp[0] =0
for i in range(0,n):
    coin.append(int(sys.stdin.readline()))
coin.sort()
for i in range(coin[0],k+1):
    for j in coin:
        if j<=i:
            dp[i] = min(dp[i],dp[i-j]+1)
if dp[k] == 10001:
    print(-1)
else:
    print(dp[k])
