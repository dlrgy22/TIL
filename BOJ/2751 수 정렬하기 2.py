import sys
n = int(sys.stdin.readline())
num_list = [int(sys.stdin.readline()) for i in range(n)]
for element in sorted(num_list):
    print(element)