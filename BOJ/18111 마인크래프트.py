import sys
import math

def blocks_info(blocks):
    num_blocks = 0
    min_height = math.inf
    max_height = 0
    for i in range(n):
        for j in range(m):
            num_blocks += blocks[i][j]
            min_height = min(min_height, blocks[i][j])
            max_height = max(max_height, blocks[i][j])
    max_height = min(max_height, (num_blocks + b)//(n*m))
    return min_height, max_height


n, m, b = map(int, sys.stdin.readline().split())
blocks = [list(map(int, sys.stdin.readline().split())) for i in range(n)]
min_height, max_height = blocks_info(blocks)
min_time = math.inf

for height in range(min_height, max_height + 1):
    time = 0
    for i in range(n):
        for j in range(m):
            if height < blocks[i][j]:
                time += (blocks[i][j] - height) * 2
            else:
                time += (height - blocks[i][j])
    if min_time >= time:
        min_time = time
        answer = height
print(min_time, answer)
