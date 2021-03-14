import sys
n = int(sys.stdin.readline())
m = int(sys.stdin.readline())
s = sys.stdin.readline()[:-1]

prev_c = ""
answer = 0
count = 0
idx = 0
while idx < m - 3:
    if s[idx: idx+3] == "IOI":
        count += 1
        idx += 2
        if count == n:
            answer += 1
            count -= 1
    else:
        idx += 1
        count = 0

print(answer)
    