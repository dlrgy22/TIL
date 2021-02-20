import sys
from collections import deque

def BFS(top, start, start_link, up, down):
    queue = deque()
    queue.appendleft([start, 0])
    visit = [False for i in range(top + 1)]
    visit[start] = True
    while len(queue) != 0:
        location, count = queue.pop()
        if location == start_link:
            return count
        next_locations = [location + up, location - down]
        for next_location in next_locations:
            if next_location <= top and next_location > 0 and not visit[next_location]:
                queue.appendleft([next_location, count + 1])
                visit[next_location] = True
    return "use the stairs"


F, S, G, U, D = map(int, 
sys.stdin.readline().split())
print(BFS(F, S, G, U, D))