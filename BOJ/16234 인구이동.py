import sys
from collections import deque

def BFS(location):
    global l, r
    check = [[1, 0], [0, 1], [-1, 0], [0, -1]]
    open = [[location[0], location[1]]]
    sum = 0
    q = deque()
    q.appendleft([location[0], location[1]])
    while len(q) != 0:
        loc = q.pop()
        for check_element in check:
            loc_y = loc[0] + check_element[0]
            loc_x = loc[1] + check_element[1]
            if loc_y < n and loc_y >= 0 and loc_x < n and loc_x >= 0 and not visit[loc_y][loc_x]:
                differ = abs(people[loc_y][loc_x] - people[loc[0]][loc[1]])
                if differ >= l and differ <= r:
                    open.append([loc_y, loc_x])
                    visit[loc_y][loc_x] = True
                    q.appendleft([loc_y, loc_x])
    for open_element in open:
        sum += people[open_element[0]][open_element[1]]
    for open_element in open:
        people[open_element[0]][open_element[1]] = sum // len(open)

def check_differ(location):
    global l, r
    check = [[1, 0], [0, 1], [-1, 0], [0, -1]]
    for check_element in check:
        loc_y = location[0] + check_element[0]
        loc_x = location[1] + check_element[1]

        if loc_y < n and loc_y >= 0 and loc_x < n and loc_x >= 0 and not visit[loc_y][loc_x]:
            differ = abs(people[loc_y][loc_x] - people[location[0]][location[1]])

            if differ >= l and differ <= r:
                return True

    return False
n, l, r = map(int, sys.stdin.readline().split())
people = [list(map(int, sys.stdin.readline().split())) for i in range(n)]
count = 0

while True:
    check = False
    visit = [[False for i in range(n)] for j in range(n)]
    for i in range(n):
        for j in range(n):
            if not visit[i][j]:
                if check_differ([i, j]):
                    check = True
                    visit[i][j] = True
                    BFS([i, j])
            visit[i][j] = True
    if not check:
        print(count)
        break
    count += 1