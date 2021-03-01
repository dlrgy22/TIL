import sys
n = int(sys.stdin.readline())
num_sum = 1
count = 1
while num_sum < n:
    num_sum += count*6
    count += 1
print(count)