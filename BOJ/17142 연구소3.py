import sys
import math
from _collections import deque
from itertools import combinations

def find_virus_wall():
    for i in range(n):
        for j in range(n):
            if virus[i][j] == 2:
                virus_location.append([i, j])
            elif virus[i][j] == 1:
                visit[i][j] = True

def BFS(idx):
    v = visit[:]
    q = deque()
    for idx_element in idx:
        v[idx_element[0]][idx_element[1]] = True
        q.appendleft([idx_element[0],idx_element[1], 0])


n, m = map(int, sys.stdin.readline().split())
virus = [list(map(int, sys.stdin.readline().split())) for i in range(n)]
virus_location = []
visit = [[False for i in range(n)] for j in range(n)]
answer = math.inf
find_virus_wall()
for element in combinations(range(len(virus_location)), m):
    BFS(element)