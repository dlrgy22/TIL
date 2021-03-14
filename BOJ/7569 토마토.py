import sys
from collections import deque

def check_tomato():
    for i in range(h):
        for j in range(n):
            for k in range(m):
                if tomato[i][j][k] == 0:
                    return False
    return True

def BFS(visit, start):
    move = [[1, 0, 0], [-1, 0, 0], [0, 1, 0], [0, -1, 0], [0, 0, 1], [0, 0, -1]]
    q = deque()
    for location in start:
        q.appendleft([[location[0], location[1], location[2]], 0])
        visit[location[0]][location[1]][location[2]] = True
    answer = -1
    while len(q) != 0:
        location, count = q.pop()
        answer = max(answer, count)
        for move_element in move:
            next_location = [location[0] + move_element[0], location[1] + move_element[1], location[2] + move_element[2]]
            if next_location[0] >= 0 and next_location[0] < h and next_location[1] >=0 and next_location[1] < n and next_location[2] >= 0 and next_location[2] < m:
                if tomato[next_location[0]][next_location[1]][next_location[2]] == 0 and not visit[next_location[0]][next_location[1]][next_location[2]]:
                    tomato[next_location[0]][next_location[1]][next_location[2]] = 1
                    visit[next_location[0]][next_location[1]][next_location[2]] = True
                    q.appendleft([next_location, count + 1])
    
    print(answer) if check_tomato() else print(-1)

m, n, h = map(int, sys.stdin.readline().split())
tomato = []
start = []
for i in range(h):
    tomato.append([list(map(int, sys.stdin.readline().split())) for i in range(n)])

visit = [list(list(False for i in range(m)) for j in range(n)) for k in range(h)]

for i in range(h):
    for j in range(n):
        for k in range(m):
            if tomato[i][j][k] == 1:
                start.append([i, j, k])
BFS(visit, start)
