import sys
from collections import deque

def DFS(graph, node, visit, answer):
    answer.append(node)
    for idx in range(1, n + 1):
        if graph[node][idx] == 1 and not visit[idx]:
            visit[idx] = True
            DFS(graph, idx, visit, answer)

def BFS(graph, node, visit, answer):
    q = deque()
    q.appendleft(node)
    while len(q) != 0:
        location = q.pop()
        answer.append(location)
        for idx in range(1, n + 1):
            if graph[location][idx] == 1 and not visit[idx]:
                visit[idx] = True
                q.appendleft(idx)

n, m, v = map(int, sys.stdin.readline().split())
graph = [[0 for i in range(n + 1)] for j in range(n + 1)]
DFS_visit = [False for i in range(n + 1)]
BFS_visit = [False for i in range(n + 1)]
DFS_visit[v] = True
BFS_visit[v] = True
for i in range(m):
    node1, node2 = map(int, sys.stdin.readline().split())
    graph[node1][node2] = 1
    graph[node2][node1] = 1

DFS_answer = []
BFS_answer = []
DFS(graph, v, DFS_visit, DFS_answer)
BFS(graph, v, BFS_visit, BFS_answer)
for element in DFS_answer:
    print(element, end = ' ')
print()
for element in BFS_answer:
    print(element, end = ' ')