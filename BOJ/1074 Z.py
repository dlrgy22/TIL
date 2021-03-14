import sys
n, r, c = map(int, sys.stdin.readline().split())
answer = 0
for i in range(n-1, -1, -1):
    num = 2**i
    if r >= num:
        r -= num
        if c >= num:
            answer += num**2*3
            c -= num
        else:
            answer += num**2*2
    else:
        if c >= num:
            answer += num**2
            c -= num
    
print(answer)