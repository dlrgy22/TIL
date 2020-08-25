import sys
import itertools
import math

def move():
    for idx in range(len(check_Enemy) - 1, -1 , -1):
        if check_Enemy[idx][0] + 1 >= n:
            del check_Enemy[idx]
        else:
            check_Enemy[idx] = [check_Enemy[idx][0] + 1, check_Enemy[idx][1]]

def manhattan_distance(Archer, Enemy):
    return abs(Archer[0] - Enemy[0]) + abs(Archer[1] - Enemy[1])

n, m ,d = map(int, sys.stdin.readline().split())
Enemy = []
castle = [list(map(int, sys.stdin.readline().split())) for i in range(n)]

for i in range(n):
    for j in range(m):
        if castle[i][j] == 1:
            Enemy.append([i, j])

answer = 0
Archer_com = list(itertools.combinations(range(m), 3))
for Archer in Archer_com:
    check_Enemy = Enemy[:]
    count = 0
    while len(check_Enemy) != 0:
        del_idx = [-1 for i in range(3)]
        del_distance = [math.inf for i in range(3)]
        for Enemy_idx in range(len(check_Enemy)):
            for Archer_idx in range(3):
                dis = manhattan_distance([n, Archer[Archer_idx]], [check_Enemy[Enemy_idx][0], check_Enemy[Enemy_idx][1]])
                if dis <= del_distance[Archer_idx] and dis <= d:
                    if dis == del_distance[Archer_idx]:
                        if check_Enemy[Enemy_idx][1] >= check_Enemy[del_idx[Archer_idx]][1]:
                            continue
                    del_idx[Archer_idx] = Enemy_idx
                    del_distance[Archer_idx] = dis
        del_idx = sorted(list(set(del_idx)), reverse = True)
        for del_element in del_idx:
            if del_element != -1:
                del check_Enemy[del_element]
                count += 1
        move()
    if answer < count:
        answer = count
print(answer)