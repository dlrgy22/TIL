import sys
import queue
def check_person(check, know_true, party):
    q = queue.Queue()
    for i in range(1, know_true[0] + 1):
        check[know_true[i]] = 1
        q.put(know_true[i])
    while q.qsize() != 0:
        c = q.get()
        for i in range(len(party)):
            if c in party[i][1:]:
                for j in range(1,party[i][0] + 1):
                    if check[party[i][j]] == 0:
                        check[party[i][j]] = 1
                        q.put(party[i][j])
n, m = map(int,sys.stdin.readline().split())
know_true = list(map(int,sys.stdin.readline().split()))
party = []
check = [0 for i in range(n + 1)]

for i in range(m):
    add = list(map(int,sys.stdin.readline().split()))
    if not len(add) == 1:
        party.append(add)
m = len(party)
check_party = [1 for i in range(m)]

check_person(check,know_true,party)

for i in range(1,n + 1):
    if check[i] == 1 :
        for j in range(m):
            if i in party[j][1:] and check_party[j] == 1:
                check_party[j] = 0

print(sum(check_party))