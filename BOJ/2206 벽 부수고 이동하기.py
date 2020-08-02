import sys
from collections import deque

def BFS():
    q = deque()
    move = [[0, 1], [1, 0], [0, -1], [-1, 0]]
    visit = [[[False, False] for i in range(m)] for j in range(n)]
    visit[0][0][0] = True
    q.appendleft([0, 0, 1, False])
    while len(q) != 0:
        element = q.pop()
        if element[0] == (n - 1) and element[1] == (m - 1):
            return element[2]
        for move_element in move:
            loc_x = element[1] + move_element[1]
            loc_y = element[0] + move_element[0]
            if loc_x >= 0 and loc_x < m and loc_y >=0 and loc_y < n:
                if map[loc_y][loc_x] == '0':
                    if element[3] and not visit[loc_y][loc_x][1]:
                        visit[loc_y][loc_x][1] = True
                        q.appendleft([loc_y, loc_x, element[2] + 1, element[3]])
                    elif not element[3] and not visit[loc_y][loc_x][0]:
                        visit[loc_y][loc_x][0] = True
                        q.appendleft([loc_y, loc_x, element[2] + 1, element[3]])

                else:
                    if not element[3] and not visit[loc_y][loc_x][1]:
                        visit[loc_y][loc_x][1] = True
                        q.appendleft([loc_y, loc_x, element[2] + 1, True])
    return -1





n, m = map(int, sys.stdin.readline().split())
map = [list(sys.stdin.readline().replace('\n', '')) for i in range(n)]
print(BFS())