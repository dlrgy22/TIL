import sys
from collections import deque

t = int(sys.stdin.readline())
for i in range(t):
    n, m = map(int, sys.stdin.readline().split())
    q_list = list(map(int, sys.stdin.readline().split()))
    q = deque()
    for idx, element in enumerate(q_list):
        q.appendleft([element, idx])
    
    count = 0
    while len(q) != 0:
        value, idx = q.pop()
        if len(q) != 0 and value < max(q)[0]:
            q.appendleft([value, idx])
        else:
            count += 1
            if idx == m:
                break
    print(count)