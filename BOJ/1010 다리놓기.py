import sys

def nCr(n, r):
    result = 1
    for i in range(n, n - r, - 1):
        result *= i
    for i in range(r, 0, -1):
        result //= i

    return result

t = int(sys.stdin.readline())
for i in range(t):
    n, m = map(int, sys.stdin.readline().split())
    print(nCr(m, n))