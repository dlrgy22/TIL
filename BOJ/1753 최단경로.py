import sys
import math
import heapq
V, E = map(int, sys.stdin.readline().split())
start = int(sys.stdin.readline())
graph = [{} for j in range(V + 1)]
visit = [False for i in range(V + 1)]
for i in range(E):
    u, v, w = map(int, sys.stdin.readline().split())
    if v in graph[u]:
        if w < graph[u][v]:
            graph[u][v] = w
    else:
        graph[u][v] = w
answer = [math.inf for i in range(V + 1)]
answer[start] = 0
hq = []
heapq.heappush(hq, [0, start])

while len(hq) != 0:
    cost, node = heapq.heappop(hq)
    for i in graph[node].keys():
        if answer[i] > cost + graph[node][i]:
            answer[i] = cost + graph[node][i]
            heapq.heappush(hq, [answer[i], i])

for i in range(1, V + 1):
    if answer[i] == math.inf:
        print('INF')
    else:
        print(answer[i])
