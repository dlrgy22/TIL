import sys
from collections import deque

def check_coin(loc_y, loc_x):
    global n, m
    if loc_x >= m or loc_x < 0 or loc_y >= n or loc_y < 0:
        return False
    return True

n, m = map(int, sys.stdin.readline().split())
board = [list(sys.stdin.readline().replace('\n', '')) for i in range(n)]
move = [[1, 0], [0, 1], [-1, 0], [0, -1]]
location = []
visit = [[[[False for i in range(m)] for j in range(n)] for k in range(m)] for l in range(n)]

for i in range(n):
    for j in range(m):
        if board[i][j] == 'o':
            location.append([i, j])
location.append(0)
q = deque()
q.appendleft(location)
visit[location[0][0]][location[0][1]][location[1][0]][location[1][1]] = True
while len(q) != 0:
    loc = q.pop()
    if loc[2] >= 10:
        break
    for move_element in move:
        check_coin1 = False
        check_coin2 = False
        coin1 = [loc[0][0] + move_element[0], loc[0][1] + move_element[1]]
        coin2 = [loc[1][0] + move_element[0], loc[1][1] + move_element[1]]
        if not check_coin(coin1[0], coin1[1]):
            check_coin1 = True
        if not check_coin(coin2[0], coin2[1]):
            check_coin2 = True

        if check_coin1 or check_coin2:
            if check_coin1 and check_coin2:
                continue
            print(loc[2] + 1)
            sys.exit()
        else:
            if board[coin1[0]][coin1[1]] == '#':
                coin1 = loc[0]
            if board[coin2[0]][coin2[1]] == '#':
                coin2 = loc[1]
            if coin1 == coin2:
                coin1 = loc[0]
                coin2 = loc[1]
            if not visit[coin1[0]][coin1[1]][coin2[0]][coin2[1]]:
                visit[coin1[0]][coin1[1]][coin2[0]][coin2[1]] = True
                q.appendleft([coin1, coin2, loc[2] + 1])

print(-1)