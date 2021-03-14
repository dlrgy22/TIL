import sys
from collections import deque

n = int(sys.stdin.readline())
queue = deque()
for i in range(n):
    command = list(sys.stdin.readline().split())
    if command[0] == "push":
        queue.appendleft(command[1])
    elif command[0] == "pop":
        print(queue.pop()) if len(queue) != 0 else print("-1")
    elif command[0] == "size":
        print(len(queue))
    elif command[0] == "empty":
        print(1) if len(queue) == 0 else print("0")
    elif command[0] == "front":
        print(queue[-1]) if len(queue) != 0 else print(-1)
    elif command[0] == "back":
        print(queue[0]) if len(queue) != 0 else print(-1)
