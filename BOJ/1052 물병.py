import sys

n, k = map(int, sys.stdin.readline().split())
bottle = []
while n // 2 != 0:
    bottle.append(n % 2)
    n //= 2
bottle.append(n % 2)

count = 0
check = True
for idx in range(len(bottle) - 1, -1, -1):
    if bottle[idx] == 1:
        count += 1
        if count == k:
            break

total = 0
sum = 0
for i in range(idx):
    if bottle[i] == 1:
        sum += 2 ** i

answer = 2 ** (idx) - sum
if sum == 0:
    answer = 0

print(answer)