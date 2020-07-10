import sys
import queue
def flood(water):
    new_water = []
    move = [[1,0],[-1,0],[0,1],[0,-1]]
    for w in water:
        for m in move:
            loc_wx= w[0] + m[0]
            loc_wy = w[1] + m[1]
            if loc_wx >= 0 and loc_wx < c and loc_wy >= 0 and loc_wy < r:
                if map[loc_wy][loc_wx] == '.' or map[loc_wy][loc_wx] == 'S':
                    map[loc_wy][loc_wx] = '*'
                    new_water.append([loc_wx,loc_wy])

    return new_water


q = queue.Queue()
r,c = map(int,sys.stdin.readline().split())
map = [list(sys.stdin.readline().replace("\n",""))for i in range(r)]
move = [[1,0],[-1,0],[0,1],[0,-1]]
visit = [[False for i in range(c)] for j in range(r)]
water = []
for i in range(r):
    for j in range(c):
        if map[i][j] == 'S':
            hedgehog = [j,i]
        elif map[i][j] == 'D':
            end_point = [j,i]
        elif map[i][j] == '*':
            water.append([j,i])
visit[hedgehog[1]][hedgehog[0]] = True
time = 0
q.put([hedgehog[0],hedgehog[1],0])


while q.qsize() != 0:
    location = q.get()
    if time == location[2]:
        time += 1
        water = flood(water)
    for m in move:
        loc_hx = location[0] + m[0]
        loc_hy = location[1] + m[1]
        if loc_hx == end_point[0] and loc_hy == end_point[1]:
            print(time)
            sys.exit()
        if loc_hx >= 0 and loc_hx < c and loc_hy >= 0 and loc_hy <r:
            if map[loc_hy][loc_hx] == '.' and visit[loc_hy][loc_hx] == False:
                visit[loc_hy][loc_hx] = True
                q.put([loc_hx, loc_hy, time])

print('KAKTUS')