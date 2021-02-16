import heapq
import sys

m, n = map(int, sys.stdin.readline().split())
input_map = [list(map(int, sys.stdin.readline().split())) for i in range(m)]
dp = [list(0 for i in range(n)) for j in range(m)]
dp[0][0] = 1
move = [[1, 0], [0, 1], [-1, 0], [0, -1]]
visit = set()
heap = []
heapq.heappush(heap, (-input_map[0][0], [0, 0], [0, 0]))

while len(heap) != 0:
    v, location, prev_location = heapq.heappop(heap)
    if location == [m, n]:
        break

    elif location != [0, 0]:
        dp[location[0]][location[1]] += dp[prev_location[0]][prev_location[1]]

    for move_element in move:
        locY = location[0] + move_element[0]
        locX = location[1] + move_element[1]
        if locX >= 0 and locX < n and locY >= 0 and locY < m and -v > input_map[locY][locX] and (tuple(location), (locY, locX)) not in visit:
            heapq.heappush(heap, (-input_map[locY][locX], [locY, locX], location))
            visit.add((tuple(location), (locY, locX)))
print(dp[m - 1][n - 1])