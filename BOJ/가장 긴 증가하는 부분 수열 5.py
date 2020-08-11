import sys
n = int(sys.stdin.readline())
num_list = list(map(int, sys.stdin.readline().split()))
dp = [num_list[0]]
save_idx = [0]
stack = []
length = 1
for i in range(1, n):
    if dp[length - 1] < num_list[i]:
        dp.append(num_list[i])
        save_idx.append(length)
        length += 1
    else:
        for idx in range(len(dp)):
            if dp[idx] >= num_list[i]:
                dp[idx] = num_list[i]
                save_idx.append(idx)
                break
print(length)

for i in range(n - 1, -1, -1):
    if save_idx[i] == length - 1:
        stack.append(num_list[i])
        length -= 1

for i in range(len(stack) - 1, - 1, -1):
    print(stack[i], end = ' ')
