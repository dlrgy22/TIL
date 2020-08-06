import sys
from collections import deque
def check(start, end):
    visit = [False for i in range(n + 1)]
    q = deque()
    q.appendleft(start)
    visit[start] = True
    while len(q) != 0:
        location = q.pop()
        for element in graph[location]:
            if element == end:
                return False

            if not visit[element]:
                visit[element] = True
                q.appendleft(element)
    return True


n = int(sys.stdin.readline())
m = int(sys.stdin.readline())

connect = []
graph = [[] for i in range(n + 1)]
for i in range(m):
    input = list(map(int, sys.stdin.readline().split()))
    if input[0] == input[1]:
        continue
    connect.append(input)
connect = sorted(connect, key = lambda x : x[2])

node = set()
result = 0
for connect_element in connect:
    if check(connect_element[0], connect_element[1]):
        graph[connect_element[0]].append(connect_element[1])
        graph[connect_element[1]].append(connect_element[0])
        node.update([connect_element[0], connect_element[1]])
        result += connect_element[2]

    if len(node) == n :
        break
print(result)