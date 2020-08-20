import sys
import math
import heapq
V, E = map(int, sys.stdin.readline().split())
start = int(sys.stdin.readline())
graph = [{} for j in range(V + 1)]
answer = [math.inf for i in range(V + 1)]
visit = [False for i in range(V + 1)]
visit[start] = True
for i in range(E):
    u, v, w = map(int, sys.stdin.readline().split())
    if v in graph[u]:
        if w < graph[u][v]:
            graph[u][v] = w
    else:
        graph[u][v] = w

answer[start] = 0
for i in range(1, V + 1):
    if i in graph[start]:
        answer[i] = graph[start][i]


for i in range(1, V + 1):
    print(answer[i])