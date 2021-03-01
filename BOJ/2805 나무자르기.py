import sys

def tree_len(trees, height, n):
    length = 0
    for tree in trees:
        if tree > height:
            length += (tree - height)
            if length >= n:
                return True
    return False

m, n = map(int, sys.stdin.readline().split())
trees = list(map(int, sys.stdin.readline().split()))
left = 0
right = max(trees)
while left <= right:
    mid = (left + right) // 2
    if tree_len(trees, mid, n):
        left = mid + 1
    else:
        right = mid -1
print(right)