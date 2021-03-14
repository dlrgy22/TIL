import sys
import sys
n = int(sys.stdin.readline())
coordinate = [list(map(int, sys.stdin.readline().split())) for i in range(n)]
coordinate = sorted(coordinate, key=lambda x : (x[1], x[0]))
for x, y in coordinate:
    print(x, y)
