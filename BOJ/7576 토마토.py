import sys
from collections import deque

def find_tomato(tomato, m, n):
    tomato_list = []
    no_tomato_list = []
    for i in range(n):
        for j in range(m):
            if tomato[i][j] == 1:
                tomato_list.append([i, j])
            elif tomato[i][j] == -1:
                no_tomato_list.append([i, j])
    return tomato_list, no_tomato_list

def bfs(tomato, tomato_list, no_tomato_list, m, n):
    move = [[1, 0], [0, 1], [-1, 0], [0, -1]]
    queue = deque()
    visit = [[False for i in range(m)] for j in range(n)]
    for no_tomato_location in no_tomato_list:
        visit[no_tomato_location[0]][no_tomato_location[1]] = True

    for tomato_location in tomato_list:
        queue.appendleft([tomato_location[0], tomato_location[1], 0])
        visit[tomato_location[0]][tomato_location[1]] = True
    
    while len(queue) != 0:
        y, x, count = queue.pop()
        for move_element in move:
            next_y = y+move_element[0]
            next_x = x+move_element[1]

            if next_y >= 0 and next_y < n and next_x >= 0 and next_x < m and tomato[next_y][next_x] == 0:
                tomato[next_y][next_x] = 1
                queue.appendleft([next_y, next_x, count + 1])
                visit[next_y][next_x] = True

    for i in range(n):
        for j in range(m):
            if not visit[i][j]:
                return -1
    return count

m, n = map(int, sys.stdin.readline().split())
tomato = [list(map(int, sys.stdin.readline().split())) for i in range(n)]
tomato_list, no_tomato_list = find_tomato(tomato, m, n)
print(bfs(tomato, tomato_list, no_tomato_list, m, n))