import sys
import math

channel = int(sys.stdin.readline())
broken_num = int(sys.stdin.readline())
if broken_num != 0:
    broken = set(map(int, sys.stdin.readline().split()))
else:
    broken = set()

answer = abs(channel - 100)
for number in range(1000001):
    num_list = list(str(number))
    exist_broken = False
    for num in num_list:
        if int(num) in broken:
            exist_broken = True
            break
    if not exist_broken:
        if answer >= abs(channel - number) + len(num_list):
            answer = abs(channel - number) + len(num_list)
print(answer)
