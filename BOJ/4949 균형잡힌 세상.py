import sys
from collections import deque

while True:
    input_string = sys.stdin.readline()[:-1]
    if input_string == ".":
        break
    stack = deque()
    answer = "yes"
    for char in input_string:
        if char == "["  or char == "(":
            stack.append(char)
        if char == "]" or char == ")":
            if len(stack) != 0 and stack[-1] == "[" and char == "]":
                stack.pop()
            elif len(stack) != 0 and stack[-1] == "(" and char == ")":
                stack.pop()
            else:
                answer = "no"
                break
    if len(stack) != 0:
        answer = "no"
    
    print(answer)