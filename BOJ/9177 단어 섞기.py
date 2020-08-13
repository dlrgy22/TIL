import sys
from collections import deque
t = int(sys.stdin.readline())
for test_case in range(t):
    check = False
    str1, str2, str3 = sys.stdin.readline().split(' ')
    idx = [0, 0, 0]
    q = deque()
    q.appendleft(idx)
    while len(q) != 0:
        idx = q.pop()
        if idx[2] == len(str2):
            check = True
            break
        if idx[0] < len(str1) and str1[idx[0]] == str3[idx[2]]:
            q.appendleft([idx[0] + 1, idx[1], idx[2] + 1])
        if idx[1] < len(str2) and str2[idx[1]] == str3[idx[2]]:
            q.appendleft([idx[0], idx[1] + 1, idx[2] + 1])

    if check:
        print("Data set {} : yes".format(test_case + 1))
    else:
        print("Data set {} : no".format(test_case + 1))