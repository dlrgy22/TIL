import sys
from collections import deque

def BFS(start, end, visit) :
    queue = deque()
    queue.appendleft([start, 0])
    visit[start] = True
    move = [-1, 1]
    while len(queue) != 0:
        location, time = queue.pop()
        teleport_location = [location]
        if location != 0:
            teleport = location * 2
            while teleport <= 100000:
                if teleport == end:
                    return time

                if not visit[teleport]:
                    visit[teleport] = True
                    queue.appendleft([teleport, time])
                    teleport_location.append(teleport)
                teleport *= 2

        for next_location in teleport_location:
            for move_elelment in move:
                idx = next_location + move_elelment
                if idx == end:
                    return time + 1
                if idx >=0 and idx <=100000 and not visit[idx]:
                    queue.appendleft([idx, time + 1])
                    visit[idx] = True


n, k = map(int, sys.stdin.readline().split())
visit = [False for i in range(100001)]
if n == k:
    print(0)
else:
    print(BFS(n, k , visit))
