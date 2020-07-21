import sys

def conversion(num):
    number = 0
    for i in range(len(num)):
        number += num[i] * (10 ** i)
    return number

def next_num(num):
    idx = len(num)
    check = True
    for i in range(len(num)):
        if num[i] == 4:
            check = False
            num[i] = 7
            idx = i
            break
    if check:
        for i in range(len(num)):
            num[i] = 4
        num.append(4)
    else:
        for i in range(idx):
            num[i] = 4


a, b = map(int, sys.stdin.readline().split())
num = [4]
count = 0
while conversion(num) <= b:
    if conversion(num) >= a:
        count += 1
    next_num(num)
print(count)