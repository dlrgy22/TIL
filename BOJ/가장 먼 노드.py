from collections import deque
import math

def make_graph(n, edge):
    graph = [[] for i in range(n + 1)]
    for start, end in edge:
        graph[start].append(end)
        graph[end].append(start)
    return graph

def solution(n, edge):
    answer = 0
    graph = make_graph(n, edge)
    large_dis = 0
    dis_list = [math.inf for i in range(n + 1)]
    q = deque()
    visit = [False for i in range(n + 1)]
    visit[1] = True
    q.append([1, 0])
    while len(q) != 0:
        node, dis  = q.popleft()
        if large_dis < dis:
            large_dis = dis
            answer = 1
        elif large_dis == dis:
            answer += 1

        for next_node in graph[node]:
            if not visit[next_node]:
                visit[next_node] = True
                q.append([next_node, dis + 1])
    print(answer)

n = 6
edge = 	[[3, 6], [4, 3], [3, 2], [1, 3], [1, 2], [2, 4], [5, 2]]
solution(n, edge)