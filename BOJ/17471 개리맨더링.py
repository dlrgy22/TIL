import sys
from itertools import combinations
import math
from _collections import deque

def check_connection(zone):
    q = deque()
    visit = [False for i in range(len(zone))]
    q.appendleft(zone[0])
    visit[0] = True
    while len(q) != 0:
        location = q.pop()
        for graph_element in graph[location]:
            for idx in range(len(zone)):
                if graph_element == zone[idx] and not visit[idx]:
                    visit[idx] = True
                    q.appendleft(zone[idx])

    if sum(visit) == len(zone):
        return True
    else:
        return False





n = int(sys.stdin.readline())
people = [0] + list(map(int, sys.stdin.readline().split()))
graph = [0]
for i in range(n):
    edge = list(map(int, sys.stdin.readline().split()))
    del edge[0]
    graph.append(edge)
result = math.inf

sum_people = sum(people)
for i in range(1, n//2 + 1):
    combin = combinations(range(1, n + 1), i)
    for combin_element in combin:
        zone1 = list(combin_element)
        zone2 = []
        s = 0
        for j in range(1, n + 1):
            if j not in zone1:
                zone2.append(j)
            else:
                s += people[j]
        differ = abs(sum_people - (2 * s))
        if differ < result:
            if check_connection(zone1) and check_connection(zone2):
                result = differ

if result == math.inf:
    print(-1)
else:
    print(result)

