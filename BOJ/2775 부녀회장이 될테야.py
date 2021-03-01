import sys

t = int(sys.stdin.readline())
for i in range(t):
    k = int(sys.stdin.readline())
    n = int(sys.stdin.readline())
    down_stairs = range(1, n + 1)
    for j in range(1, k + 1):
        stairs = []
        for k in range(1, n + 1):
            if k == 1:
                stairs.append(down_stairs[0])
            else:
                stairs.append(stairs[-1] + down_stairs[k - 1])
        down_stairs = stairs
    print(stairs[-1])