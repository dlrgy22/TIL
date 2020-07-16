import sys
import queue
def bfs(map, visit):
    q = queue.Queue()
    count = 0
    for i in range(n):
        for j in range(n):
            if visit[i][j] == False:
                visit[i][j] = True
                check_color = map[i][j]
                q.put([j, i])
                while q.qsize() != 0:
                    loc = q.get()
                    for m in move:
                        if loc[0] + m[0] < n and loc[0] + m[0] >= 0 and loc[1] + m[1] < n and loc[1] + m[1] >= 0 and \
                                map[loc[1] + m[1]][loc[0] + m[0]] == check_color and not visit[loc[1] + m[1]][
                            loc[0] + m[0]]:
                            q.put([loc[0] + m[0], loc[1] + m[1]])
                            visit[loc[1] + m[1]][loc[0] + m[0]] = True
                count += 1
    return count
n = int(sys.stdin.readline())
move = [[1, 0], [0, 1], [-1, 0], [0, -1]]
map = []
for i in range(n):
    map.append(list(sys.stdin.readline().replace('\n',"")))
visit = [[False for i in range(n)] for j in range(n)]
print(bfs(map, visit), end = ' ')

for i in range(n):
    for j in range(n):
        if map[i][j] == 'R':
            map[i][j] = 'G'
visit = [[False for i in range(n)] for j in range(n)]
print(bfs(map,visit))