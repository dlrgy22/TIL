import sys
from collections import deque

def Password(Key_logger):
    left = []
    right = []
    for i in range(len(Key_logger)):
        if Key_logger[i] == '<':
            if not len(left) == 0:
                right.append(left.pop())
        elif Key_logger[i] == '>':
            if not len(right) == 0:
                left.append(right.pop())
        elif Key_logger[i] == '-':
            if not len(left) == 0:
                left.pop()
        else:
            left.append(Key_logger[i])
    left.extend(reversed(right))
    print(''.join(left))


t = int(sys.stdin.readline())
for i in range(t):
    Key_logger = sys.stdin.readline().replace('\n','')
    Password(Key_logger)