import sys
n = int(sys.stdin.readline())
crane_list = list(map(int, sys.stdin.readline().split()))
m = int(sys.stdin.readline())
cargo_list = list(map(int, sys.stdin.readline().split()))
crane_list = sorted(crane_list, reverse = True)
cargo_list = sorted(cargo_list, reverse = True)
time = 0
while len(cargo_list) != 0:
    del_idx = []
    start = 0
    for crane_element in crane_list:
        for idx in range(start, len(cargo_list)):
            if crane_element >= cargo_list[idx]:
                del_idx.append(idx)
                start = idx + 1
                break
    if len(del_idx) == 0:
        time = -1
        break
    for index in range(len(del_idx) - 1, - 1, - 1):
        del cargo_list[del_idx[index]]
    time += 1

print(time)