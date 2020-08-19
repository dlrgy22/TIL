import sys
import math
def move_endpoint():
    global sum, end
    while sum < s and end < n:
        sum += num_list[end]
        end += 1

def move_startpoint():
    global sum, start
    while sum >= s:
        sum -= num_list[start]
        start += 1

n, s = map(int, sys.stdin.readline().split())
num_list = list(map(int, sys.stdin.readline().split()))
start = 0
end = 0
sum = 0
result = math.inf

while end <= n - 1:
    move_endpoint()
    if sum >= s:
        result = min(result, end - start)
    move_startpoint()
    if sum + num_list[start - 1] >= s:
        result = min(result, end - start + 1)
if result == math.inf:
    print(0)
else:
    print(result)