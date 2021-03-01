import sys
from collections import deque

n = int(sys.stdin.readline())
cards = range(1, n+1)
q = deque()
for card in cards:
    q.appendleft(card)

while len(q) != 1:
    card = q.pop()
    card = q.pop()
    q.appendleft(card)

print(q.pop())
