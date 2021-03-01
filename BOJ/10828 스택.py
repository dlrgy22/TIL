import sys
from collections import deque

n = int(sys.stdin.readline())
stack = deque()
for i in range(n):
    command = list(sys.stdin.readline().split())
    if command[0] == "push":
        stack.append(command[1])

    elif command[0] == "pop":
        print(stack.pop()) if len(stack) != 0 else print("-1")

    elif command[0] == "size":
        print(len(stack))

    elif command[0] == "empty":
        print("1") if len(stack) == 0 else print("0")
        
    else:
        print(stack[-1]) if len(stack) != 0 else print("-1")