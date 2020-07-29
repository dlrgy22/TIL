import sys
from collections import deque

def move_wall():
    remove =[]
    for i in range(len(wall)):
        wall[i][1] += 1
        if wall[i][1] == 8:
            remove.append([wall[i][0], wall[i][1]])
    for element in remove:
        wall.remove([element[0], element[1]])


def check_wall(loc):
    if loc in wall or [loc[0], loc[1] - 1] in wall:
        return False
    return True

def BFS():
    move = [[1, 0], [-1, 0], [0, 1], [0, -1], [1, 1], [1, -1], [-1, 1],[-1, -1]]
    q = deque()
    q.appendleft([0, 7, 0])
    time = 0
    while len(q) != 0:
        print(wall)
        location = q.pop()
        if location[0] == 7 and location[1] == 0:
            return(1)
        if time != location[2]:
            move_wall()
            time = location[2]

        for move_element in move:
            loc_x = location[0] + move_element[0]
            loc_y = location[1] + move_element[1]
            if loc_x >=0 and loc_x < 8 and loc_y >= 0 and loc_y < 8:
                if check_wall([loc_x, loc_y]):
                    q.appendleft([loc_x, loc_y, time + 1])
    return(0)

map = [list(sys.stdin.readline().replace('\n', '')) for i in range(8)]
wall = []
for i in range(8):
    for j in range(8):
        if map[i][j] == '#':
            wall.append([j, i])
print(BFS())



