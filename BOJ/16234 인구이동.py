import sys
import queue

def BFS(visit, loc):
    move = [[1, 0], [-1, 0], [0, 1], [0, -1]]
    save_location = [[loc[0], loc[1]]]
    sum = Population[loc[0]][loc[1]]
    q = queue.Queue()
    q.put([loc[0], loc[1]])
    while q.qsize() != 0:
        location = q.get()
        for move_element in move:
            loc_x = location[1] + move_element[1]
            loc_y = location[0] + move_element[0]
            if loc_x >= 0 and loc_x < n and loc_y >= 0 and loc_y < n:
                if not visit[loc_y][loc_x]:
                    dif = abs(Population[location[0]][location[1]] - Population[loc_y][loc_x])
                    if dif >= L and dif <= R:
                        visit[loc_y][loc_x] = True
                        sum += Population[loc_y][loc_x]
                        q.put([loc_y, loc_x])
                        save_location.append([loc_y, loc_x])
    return save_location, sum

def Population_Migration():
    visit = [[False for i in range(n)] for j in range(n)]
    check = False
    for i in range(n):
        for j in range(n):
            if not visit[i][j]:
                visit[i][j] = True
                migration, sum = BFS(visit, [i, j])
                length = len(migration)
                if length > 1:
                    check = True
                    for migration_element in migration:
                        Population[migration_element[0]][migration_element[1]] = sum // length
    return check


n, L, R = map(int, sys.stdin.readline().split())
Population = [list(map(int, sys.stdin.readline().split())) for i in range(n)]
count = 0
while Population_Migration():
    count += 1
print(count)