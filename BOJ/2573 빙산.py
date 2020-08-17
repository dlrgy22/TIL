import sys
from collections import deque
def Next_Year():
    visit = [[False for i in range(m)] for j in range(n)]
    move = [[1, 0], [0, 1], [-1, 0], [0, -1]]
    melt = [[0 for i in range(m)] for j in range(n)]
    check = False
    q = deque()
    for i in range(n):
        for j in range(m):
            if North_Pole[i][j] != 0 and not visit[i][j]:
                if not check:
                    check = True
                    q.appendleft([i, j])
                    visit[i][j] = True
                    for element in move:
                        if North_Pole[i + element[1]][j + element[0]] == 0:
                            melt[i][j] -= 1
                    while len(q) != 0:
                        location = q.pop()
                        for move_element in move:
                            loc_x = location[1] + move_element[1]
                            loc_y = location[0] + move_element[0]
                            if loc_x >= 0 and loc_x < m and loc_y >= 0 and loc_y < n:
                                if not visit[loc_y][loc_x] and North_Pole[loc_y][loc_x] != 0:
                                    visit[loc_y][loc_x] = True
                                    for element in move:
                                        if North_Pole[loc_y + element[0]][loc_x + element[1]] == 0:
                                            melt[loc_y][loc_x] -= 1
                                    q.appendleft([loc_y, loc_x])
                else:
                    return -1
    if check:
        for i in range(n):
            for j in range(m):
                North_Pole[i][j] += melt[i][j]
                if North_Pole[i][j] < 0:
                    North_Pole[i][j] = 0
        return 1
    else:
        return 0


n, m = map(int, sys.stdin.readline().split())
North_Pole = [list(map(int, sys.stdin.readline().split())) for i in range(n)]
count = 0

while True:
    c = Next_Year()
    if c == 1:
        count += 1
    elif c == -1:
        break
    else:
        count = 0
        break


print(count)