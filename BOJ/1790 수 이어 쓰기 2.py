import sys
n, m = map(int, sys.stdin.readline().split())
count = 0
for i in range(1, n + 1):
    number = str(i)
    count += len(number)
    if count >= m:
        print(number[len(number) - 1 - count + m])
        sys.exit()
print(-1)