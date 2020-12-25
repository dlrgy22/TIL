import sys
import math
n, m = map(int, sys.stdin.readline().split())
day = list(map(int, sys.stdin.readline().split()))

KOI = [True for i in range(n + 1)]
for idx in day:
    KOI[idx] = False

money = [[math.inf for i in range(n + 1)] for j in range(n + 5)]
money[0][0] = 0

for i in range(0, n):
    for j in range(n - 2):
        if money[i][j] == math.inf:
            continue

        if not KOI[i + 1]:
            money[i + 1][j] = min(money[i + 1][j], money[i][j])
            continue

        if j >= 3:
            money[i + 1][j - 3] = min(money[i][j], money[i + 1][j - 3])

        money[i + 1][j] = min(money[i + 1][j], money[i][j] + 10000)

        for k in range(1, 4):
            money[i + k][j + 1] = min(money[i + k][j + 1], money[i][j] + 25000)

        for k in range(1, 6):
            money[i + k][j + 2] = min(money[i + k][j + 2], money[i][j] + 37000)


print(min(money[n]))