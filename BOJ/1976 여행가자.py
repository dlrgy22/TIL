import sys
from collections import deque


def BFS(start_point):
    visit[start_point] = True
    q = deque()
    q.appendleft(start_point)
    while len(q) != 0:
        location = q.pop()
        for i in range(n):
            if graph[location][i] == 1 and not visit[i]:
                visit[i] = True
                q.appendleft(i)




n = int(sys.stdin.readline())
m = int(sys.stdin.readline())
graph = [list(map(int, sys.stdin.readline().split())) for i in range(n)]

travel = list(set(map(int, sys.stdin.readline().split())))

start_point = travel[0] - 1
visit = [False for i in range(n)]

BFS(start_point)

for travel_element in travel:
    if not visit[travel_element - 1]:
        print("NO")
        sys.exit()
print("YES")