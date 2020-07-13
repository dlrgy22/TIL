import sys
min, max = map(int ,sys.stdin.readline().split())
check = [1 for i in range(min, max + 1, 1)]
for i in range(2,int(max ** 0.5) + 1):
    square = i ** 2
    j = min // square
    while square * j <= max:
        if square * j - min < 0:
            j += 1
            continue
        check[square * j - min] = 0
        j += 1

print(sum(check))