import sys
from collections import deque

def BFS(start):
    q = deque()
    visit[start] = 1
    q.appendleft(start)
    while len(q) != 0:
        location = q.pop()
        for element in edge[location]:
            if visit[location] == 1:
                if visit[element] == 0:
                    visit[element] = 2
                    q.appendleft(element)
                elif visit[element] == 1:
                    return False
            elif visit[location] == 2:
                if visit[element] == 0:
                    visit[element] = 1
                    q.appendleft(element)
                elif visit[element] == 2:
                    return False
    return True


t = int(sys.stdin.readline())
for i in range(t):
    v, e = map(int, sys.stdin.readline().split())
    visit = [0 for j in range(v + 1)]
    edge = [[] for j in range(v + 1)]
    check = True
    for j in range(e):
        e1, e2 = map(int, sys.stdin.readline().split())
        edge[e1].append(e2)
        edge[e2].append(e1)

    for j in range(1, v):
        if visit[j] == 0:
            if not BFS(j):
                check = False
                break

    if check:
        print("YES")
    else:
        print("NO")