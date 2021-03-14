import sys
n, k = map(int, sys.stdin.readline().split())
num_list = list(range(1, n+1))
print("<", end='')
idx = 0
while len(num_list) != 0:
    idx += (k - 1)
    if idx > len(num_list) - 1:
        idx = idx%len(num_list)
    print(f"{num_list[idx]},", end=" ") if len(num_list) != 1 else print(num_list[idx], end="")
    del num_list[idx]
print(">")