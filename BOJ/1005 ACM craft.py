import sys

def check_order():
    for i in range(k):
        f, l = map(int, sys.stdin.readline().split())
        order[l].append(f)

def construct_time(building_number):
    if visit[building_number] != 0:
        return visit[building_number]
    large_time = 0
    for element in order[building_number]:
        tmp = construct_time(element)
        if large_time < tmp:
            large_time = tmp
    visit[building_number] = large_time + time[building_number]
    return visit[building_number]

t = int(sys.stdin.readline())
for i in range(t):
    n, k = map(int,sys.stdin.readline().split())
    time = [0] + list(map(int, sys.stdin.readline().split()))
    visit = [0 for i in range(n + 1)]
    order = [[] for i in range(n + 1)]
    check_order()
    w = int(sys.stdin.readline())
    print(construct_time(w))