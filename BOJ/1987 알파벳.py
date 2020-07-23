import sys

def DFS(location, count):
    global result
    if result < count:
        result = count
    move = [[1, 0], [0, 1], [-1, 0], [0, -1]]
    for move_element in move:
        loc_r = location[0] + move_element[0]
        loc_c = location[1] + move_element[1]
        if loc_r >= 0 and loc_r < r and loc_c >= 0 and loc_c < c:
            if not alphabet[ord(arr[loc_r][loc_c]) - ord('A')]:
                alphabet[ord(arr[loc_r][loc_c]) - ord('A')] = True
                DFS([loc_r, loc_c], count + 1)
                alphabet[ord(arr[loc_r][loc_c]) - ord('A')] = False


r, c = map(int, sys.stdin.readline().split())
arr = [list(sys.stdin.readline().replace('\n', '')) for i in range(r)]
alphabet = [False for i in range(26)]
result = 0
alphabet[ord(arr[0][0]) - ord('A')] = True
DFS([0, 0], 1)
print(result)