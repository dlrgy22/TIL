import sys
import queue
import itertools
import copy
bottle = list(map(int,sys.stdin.readline().split()))
arr = [0,1,2]
choose = list(itertools.permutations(arr,3))
water = [0,0,bottle[2]]
new_water = [0,0,bottle[2]]
check = []
result = []
q = queue.Queue()
q.put(water)
check.append(water)
while q.qsize() != 0:
    water = copy.deepcopy(q.get())
    if water[0] == 0:
        result.append(water[2])
    for i in choose:
        s = water[i[0]] + water[i[1]]
        if s > bottle[i[1]]:
            new_water[i[1]] = bottle[i[1]]
            new_water[i[0]] = s - bottle[i[1]]
            new_water[i[2]] = water[i[2]]
        else:
            new_water[i[1]] = s
            new_water[i[0]] = 0
            new_water[i[2]] = water[i[2]]
        if new_water not in check:
            q.put([new_water[0],new_water[1],new_water[2]])
            check.append([new_water[0],new_water[1],new_water[2]])
for i in sorted(result):
    print(i,end = ' ')
