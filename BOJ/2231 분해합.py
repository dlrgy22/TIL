import sys
from collections import defaultdict

n = int(sys.stdin.readline())
dic = defaultdict(list)
for i in range(1, 1000001):
    decomposition_sum = i
    for j in str(i):
        decomposition_sum += int(j)
    dic[decomposition_sum].append(i)

if n in dic:
    print(min(dic[n]))
else:
    print(0)