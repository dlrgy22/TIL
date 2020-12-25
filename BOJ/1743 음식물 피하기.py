import sys
from collections import deque

n, m, r = map(int, sys.stdin.readline().split())
trash = [[0 for i in range(m + 1)] for j in range(n + 1)]
move = [[0, 1], [1, 0], [-1, 0], [0, -1]]
for i in range(r):
    y, x = map(int, sys.stdin.readline().split())
    trash[y][x] = 1

visit = [[False for i in range(m + 1)] for j in range(n + 1)]

queue = deque()
answer = 0
for i in range(1, n + 1):
    for j in range(1, m + 1):
        if not visit[i][j] and trash[i][j] == 1:
            queue.appendleft([i, j])
            visit[i][j] = True
            size = 1
            while len(queue) != 0:
                y, x = queue.pop()
                for move_element in move:
                    next_x = x + move_element[1]
                    next_y = y + move_element[0]
                    if next_x >= 1 and next_x <= m and next_y >= 1 and next_y <= n and trash[next_y][next_x] == 1 and not visit[next_y][next_x]:
                        queue.appendleft([next_y, next_x])
                        visit[next_y][next_x] = True
                        size += 1
            answer = max(answer, size)

print(answer)