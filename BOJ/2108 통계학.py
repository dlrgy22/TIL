import sys
import math
from collections import Counter

n = int(sys.stdin.readline())
num_list = [int(sys.stdin.readline()) for i in range(n)]
cnt_list = sorted(list(Counter(num_list).items()), key=lambda x : (-x[1], x[0]))

mean = round(sum(num_list) / len(num_list))
middle = sorted(num_list)[len(num_list) // 2]
try:
    cnt = cnt_list[0][0] if cnt_list[0][1] != cnt_list[1][1] else cnt_list[1][0]
except:
    cnt = cnt_list[0][0]
range_length = max(num_list) - min(num_list)


print(mean)
print(middle)
print(cnt)
print(range_length)