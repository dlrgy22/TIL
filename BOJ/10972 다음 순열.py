import sys
import math
n = int(sys.stdin.readline())
number = list(map(int,sys.stdin.readline().split()))
idx = 0
for i in range(n - 1, 0, -1):
    if number[i] > number[i - 1]:
        idx = i
        break
if idx == 0:
    print(-1)
else:
    num = number[idx - 1]
    save_num = math.inf
    check_num =number[idx - 1:]
    for i in check_num:
        if save_num > i and num < i :
            save_num = i
    check_num.remove(save_num)
    number[idx - 1] = save_num
    for i in range(idx):
        print(number[i], end = ' ')
    check_num.sort()
    for i in check_num:
        print(i, end = ' ')