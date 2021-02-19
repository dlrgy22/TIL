import sys
import math

def direct_cost(n, m):
    cost_map = [[math.inf for i in range(n + 1)] for j in range(n + 1)]
    for i in range(m):
        start, end, cost = map(int, sys.stdin.readline().split())
        cost_map[start][end] = min(cost, cost_map[start][end])
    
    return cost_map

def answer(cost_map, n, m):
    for k in range(1, n+1):
        for i in range(1, n+1):
            for j in range(1, n+1):
                if i == j:
                    continue
                cost_map[i][j] = min(cost_map[i][j], cost_map[i][k] + cost_map[k][j])
    
    for i in range(1, n+1):
        for j in range(1, n+1):
            if cost_map[i][j] == math.inf:
                cost_map[i][j] = 0
            print(cost_map[i][j], end=' ') if j != n else print(cost_map[i][j])
            

n = int(sys.stdin.readline())
m = int(sys.stdin.readline())
cost_map = direct_cost(n, m)
answer(cost_map, n ,m)


