import sys
n = int(sys.stdin.readline())
meetings = sorted([list(map(int, sys.stdin.readline().split())) for i in range(n)], key = lambda x : (x[1], x[0]))

time = 0
answer = 0
for meeting in meetings:
    if time <= meeting[0]:
        time = meeting[1]
        answer += 1
print(answer)
