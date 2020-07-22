import sys
from collections import deque

def BFS(location):
    move = [[1, 0], [0, 1], [-1, 0], [0, -1]]
    visit = [[False for i in range(x)] for j in range(y)]
    distance = 0
    q.append([location[0], location[1], 0])
    visit[location[0]][location[1]] = True
    while q:
        loc = q.popleft()
        for move_element in move:
            loc_y = loc[0] + move_element[0]
            loc_x = loc[1] + move_element[1]
            if loc_x >= 0 and loc_x < x and loc_y >= 0 and loc_y < y:
                if map[loc_y][loc_x] == 'L' and not visit[loc_y][loc_x]:
                    distance = loc[2] + 1
                    visit[loc_y][loc_x] = True
                    q.append([loc_y, loc_x, loc[2] + 1])
    return distance


y, x = map(int, sys.stdin.readline().split())
q = deque()
map = [list(sys.stdin.readline().replace('\n', '')) for i in range(y)]
max_distance = 0
for i in range(y):
    for j in range(x):
        if map[i][j] == 'L':
            max_distance = max(BFS([i, j]), max_distance)
print(max_distance)