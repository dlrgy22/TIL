import sys
from collections import deque
def find_wall():
    for i in range(8):
        for j in range(8):
            if  chess_board[i][j] == '#':
                wall.append([i, j])

def find_next_wall():
    for wall_element in wall:
        if wall_element[0] + 1 < 8:
            next_wall.append([wall_element[0] + 1, wall_element[1]])

def BFS():
    global wall, next_wall
    move = [[0, 0], [1, 0],  [0, 1], [-1, 0], [0, -1], [1, 1], [1, -1], [-1, 1], [-1, -1]]
    time = 0
    q = deque()
    q.appendleft([7, 0, 0])
    find_next_wall()
    while len(q) != 0:
        location = q.pop()

        if time != location[2]:
            time += 1
            wall = next_wall
            next_wall = []
            find_next_wall()

        for move_element in move:
            loc_y = location[0] + move_element[0]
            loc_x = location[1] + move_element[1]
            if loc_x >= 0 and loc_x < 8 and loc_y >= 0  and loc_y < 8:
                if loc_y == 0 and  loc_x == 7:
                    return 1
                if [loc_y, loc_x] not in wall and [loc_y, loc_x] not in next_wall:
                    q.appendleft([loc_y, loc_x, location[2] + 1])
    return 0

chess_board = [list(sys.stdin.readline().replace('\n', '')) for i in range(8)]
wall = []
next_wall = []
find_wall()
start_point = [7, 0]
end_point = [0, 7]
print(BFS())
