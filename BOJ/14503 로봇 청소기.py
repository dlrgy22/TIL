import sys
n,m = map(int,sys.stdin.readline().split())
y,x,dir = map(int,sys.stdin.readline().split())
arr = [list(map(int,sys.stdin.readline().split())) for i in range(n)]
result = 0
while True:
    if arr[y][x] == 0:
        result += 1
        arr[y][x] = 2
        count = 0
    else:
        count = 0
        for i in range(4):
            dir -= 1
            if dir == -1:
                dir = 3
            if dir == 0 and arr[y - 1][x] == 0:
                y -= 1
                break
            elif dir == 1 and arr[y][x + 1] == 0:
                x += 1
                break
            elif dir == 2 and arr[y + 1][x] == 0:
                y += 1
                break
            elif dir == 3 and arr[y][x - 1] == 0:
                x -= 1
                break
            count += 1
    if count == 4:
        if dir == 0:
            if arr[y + 1][x] == 1:
                break
            else:
                y += 1
        elif dir == 1:
            if arr[y][x - 1] == 1:
                break
            else:
                x -= 1
        elif dir == 2:
            if arr[y - 1][x] == 1:
                break
            else:
                y -= 1
        elif dir == 3:
            if arr[y][x + 1] == 1:
                break
            else:
                x += 1
print(result)