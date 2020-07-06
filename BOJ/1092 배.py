import sys
n = int(sys.stdin.readline())
c_weight = list(map(int,sys.stdin.readline().split()))
c_weight.sort()
count = [0 for i in range(len(c_weight))]
result = 0
m = int(sys.stdin.readline())
b_weight = list(map(int,sys.stdin.readline().split()))
total_box = len(b_weight)
for i in b_weight:
    for j in range(len(c_weight)):
        if i < c_weight[j] :
            count[j] += 1
for i in count:
    total_box -= i * len(c_weight)
    