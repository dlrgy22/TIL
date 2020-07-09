import sys
def check_count(dis,house):
    check_point = 0
    cnt = 1
    for i in range(n):
        if house[i] - house[check_point] >= dis:
            cnt += 1
            check_point = i
    return cnt
n, c = map(int,sys.stdin.readline().split())
house = []
for i in range(n):
    house.append(int(sys.stdin.readline()))
house.sort()
start = 1
end = house[n - 1] - house[0]

while start <= end:
    mid = (start + end) // 2
    count = check_count(mid,house)
    if count >= c:
        result = mid
        start = mid + 1
    elif count < c:
        end = mid - 1

print(result)
