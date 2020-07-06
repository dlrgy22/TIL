import sys
n = int(sys.stdin.readline())
tri = []
for i in range(n):
    tri.append(list(map(int,sys.stdin.readline().split())))
for i in range(n - 1, -1 ,-1):
    for j in range(len(tri[i]) - 1):
        tri[i - 1][j] += max(tri[i][j],tri[i][j + 1])
print(tri[0][0])