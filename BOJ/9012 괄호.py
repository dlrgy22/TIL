import sys
from collections import deque

n = int(sys.stdin.readline())
for i in range(n):
    string = sys.stdin.readline()[:-1]
    stack = deque()
    answer = "YES"
    for char in string:
        if char == "(":
            stack.append(char)
        else:
            if len(stack) == 0:
                answer = "NO"
                break
            else:
                stack.pop()
    if len(stack) != 0:
        answer = "NO"
    print(answer)
            