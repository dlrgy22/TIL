import sys
import re
from collections import deque

t = int(sys.stdin.readline())
r = re.compile(",|\[|\]")
for i in range(t):
    back = False
    functions = sys.stdin.readline()[:-1]
    n = int(sys.stdin.readline())
    input_list = sys.stdin.readline()[:-1]
    input_list = list(map(int, r.sub(" ", input_list).split()))
    for function in functions:
        if function == "R":
            back = not back
        else:
            try:
                if back:
                    del input_list[-1]
                else:
                    del input_list[0]
            except:
                input_list = "error"
                break
    if input_list == "error":
        print("error")
        continue
    if back:
        input_list = input_list[::-1]
    print("[", end='')
    for idx in range(len(input_list)):
        print(input_list[idx], end='')
        if idx != len(input_list) - 1:
            print(",", end="")
    print("]")
