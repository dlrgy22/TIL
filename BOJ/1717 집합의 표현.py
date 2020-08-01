import sys

def parent_num(num):
    if num_set[num] == num:
        return num
    p_num = parent_num(num_set[num])
    num_set[num] = p_num
    return p_num

def sum_oper(a, b):
    a = parent_num(a)
    b = parent_num(b)

    if a != b:
        num_set[b] = a

n, m = map(int, sys.stdin.readline().split())
num_set = {}

for i in range(n + 1):
    num_set[i] = i

for i in range(m):
    oper, a, b = map(int, sys.stdin.readline().split())
    if oper == 0 :
        sum_oper(a, b)
    else:
        if parent_num(a) == parent_num(b):
            print("YES")
        else:
            print("NO")