import sys
import itertools
import math
def check_chicken_distance(house,chiken,index):
    chicken_distance = [math.inf for i in range(len(house))]
    for i in index:
        for j in range(len(house)):
            chicken_distance[j] = min(manhattan_distance(chiken[i],house[j]), chicken_distance[j])
    return(sum(chicken_distance))

def manhattan_distance(loc1, loc2):
    return (abs(loc1[0] - loc2[0]) + abs(loc1[1] - loc2[1]))

n, m = map(int, sys.stdin.readline().split())
map = [list(map(int, sys.stdin.readline().split())) for i in range(n)]
result = math.inf
chicken = []
house = []
for i in range(n):
    for j in range(n):
        if map[i][j] == 1:
          house.append([j,i])
        elif map[i][j] == 2:
            chicken.append([j,i])

index_list = list(itertools.combinations(list(range(len(chicken))),m))
for index in index_list:
    result = min(check_chicken_distance(house,chicken,index), result)

print(result)