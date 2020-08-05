import sys
import math
def increase_endpoint():

    global result, sum, start, end
    while sum + num_list[end] <= s:

        sum += num_list[end]
        end += 1
        if end == n:
            break

    if sum == s:
        result = min(result, (end - start))
    if end < n:
        sum += num_list[end]
        end += 1


def increase_startpoint():

    global result, sum, start, end
    while sum - num_list[start] >= s and start != end:

        sum -= num_list[start]
        start += 1

    if sum == s:
        result = min(result, (end - start))
    sum -= num_list[start]
    start += 1


n, s = map(int ,sys.stdin.readline().split())
num_list = list(map(int, sys.stdin.readline().split()))
start = 0
end = 0
sum = 0
result = math.inf

increase_endpoint()
while end <= n - 1:
    increase_startpoint()
    increase_endpoint()
increase_startpoint()

if result == math.inf:
    result = 0
print(result)
