import sys

def find_parent(edge):
    if parent[edge] == edge:
        return edge
    else:
        p_edge = find_parent(parent[edge])
        parent[edge] = p_edge
        return p_edge

v, e = map(int, sys.stdin.readline().split())
edges = []
parent = [i for i in range(v + 1)]

for i in range(e):
    edge = list(map(int, sys.stdin.readline().split()))
    edges.append(edge)

edges = sorted(edges, key = lambda x : x[2])
result = 0
count = 0

for edges_element in edges:
    P_start = find_parent(edges_element[0])
    P_end = find_parent(edges_element[1])
    if P_start != P_end:
        parent[P_end] = P_start
        result += edges_element[2]
        count += 1
    if count == v - 1:
        break

print(result)