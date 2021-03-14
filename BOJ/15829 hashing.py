import sys
L = int(sys.stdin.readline())
string = sys.stdin.readline()[:-1]
answer = 0
for idx, element in enumerate(string):
    alpa = ord(string[idx]) - ord("a") + 1
    answer += alpa*(31**idx)
print(answer % 1234567891)
