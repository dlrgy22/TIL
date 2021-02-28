import sys
import math

def line_count(lines,length):
    count = 0
    for line in lines:
        count += line // length
    return count


k, n = map(int, sys.stdin.readline().split())
lans = [int(sys.stdin.readline()) for i in range(k)]
left = 1
right = max(lans)

while left <= right:
    mid = (left + right) // 2
    count = line_count(lans, mid)
    if count >= n:
        left = mid + 1
    else:
        right = mid - 1
print(right)