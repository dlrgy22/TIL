import sys
import math
import copy
from _collections import deque
from itertools import combinations

def print_v(v):
    for i in v:
        for j in i:
            print(j, end = ' ')
        print()
    print()


def visit_area(v):
    number = 0
    for element in v:
        number += sum(element)
    return number

def find_virus_wall():
    for i in range(n):
        for j in range(n):
            if virus[i][j] == 2:
                virus_location.append([i, j])
            elif virus[i][j] == 1:
                visit[i][j] = True

def BFS(idx):
    global n
    move = [[1, 0], [0, 1], [-1, 0], [0, -1]]
    v = copy.deepcopy(visit)
    q = deque()
    time = 0
    for idx_element in idx:
        v[virus_location[idx_element][0]][virus_location[idx_element][1]] = True
        q.appendleft([virus_location[idx_element][0], virus_location[idx_element][1], 0])
    while len(q) != 0:
        location = q.pop()
        time = location[2]
        for move_element in move:
            loc_y = location[0] + move_element[0]
            loc_x = location[1] + move_element[1]
            if loc_x >= 0 and loc_x < n and loc_y >= 0 and loc_y < n:
                if not v[loc_y][loc_x]:
                    if virus[loc_y][loc_x] == 2:
                        q.append([loc_y, loc_x, location[2]])
                        v[loc_y][loc_x] = True
                    else:
                        q.appendleft([loc_y, loc_x, location[2] + 1])
                        v[loc_y][loc_x] = True


    if visit_area(v) == (n * n):

        return time
    else:
        return math.inf
n, m = map(int, sys.stdin.readline().split())
virus = [list(map(int, sys.stdin.readline().split())) for i in range(n)]
virus_location = []
visit = [[False for i in range(n)] for j in range(n)]
answer = math.inf
find_virus_wall()
for element in combinations(range(len(virus_location)), m):
    time = BFS(element)
    answer = min(time, answer)


if answer == math.inf:
    print(-1)
else:
    print(answer)