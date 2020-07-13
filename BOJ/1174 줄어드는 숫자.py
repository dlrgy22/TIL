import sys
import itertools

n = int(sys.stdin.readline())
number = list(range(9, -1, -1))
count = 0
for i in range(1,11):
    combination = list(itertools.combinations(number, i))
    for j in range(len(combination) -1, -1, -1):
        count += 1
        if count == n:
            for p in combination[j]:
                print(p, end = '')
            sys.exit()
print(-1)