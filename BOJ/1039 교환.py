import sys

def swap(string ,idx1, idx2):
    tmp = string[idx1]
    string[idx1] = string[idx2]
    string[idx2] = tmp

n, k = sys.stdin.readline().split()
n = list(n)

n_list = []
for i in range(len(n)):
    n_list.append([n[i], i])

n_list = sorted(n_list, reverse = True)

count = 0

for i in range(len(n_list)):
    if n[i] < n_list[i][0]:
        swap(n, i, n_list[i][1])
        count += 1
        if count == int(k):
            break

if (int(k) - count)  % 2 == 1:
    swap(n, len(n)- 2, len(n) -1)

if n[0] == '0':
    print(-1)

else:
    for i in range(len(n)):
        print(n[i], end ='')