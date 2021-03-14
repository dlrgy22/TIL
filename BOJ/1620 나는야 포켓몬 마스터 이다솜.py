import sys
n, m = map(int, sys.stdin.readline().split())
monster2num = dict()
num2monster = dict()

for i in range(1, n + 1):
    monster = sys.stdin.readline()[:-1]
    monster2num[monster] = str(i)
    num2monster[str(i)] = monster
for i in range(m):
    data = sys.stdin.readline()[:-1]
    print(monster2num[data]) if data in monster2num else print(num2monster[data])