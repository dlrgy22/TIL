import sys
n, m = map(int, sys.stdin.readline().split())
arr = [list(sys.stdin.readline().replace('\n','')) for i in range(n)]
check = [[0 for j in range(m)]for i in range(n)]
result = 0
for i in range(n):
    for j in range(m):
        if arr[i][j] == '1':
            if i > 0 and j > 0:
                length =  min(check[i][j - 1], check[i - 1][j], check[i - 1][j -1])
                check[i][j] = length + 1
                result = max(result, check[i][j])
            else:
                check[i][j] = 1
                result = max(result, 1)

print(result ** 2)