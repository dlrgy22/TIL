import sys
n = int(sys.stdin.readline())
count = 0
number = 665
while count != n:
    number += 1
    if "666" in str(number):
        count += 1
print(number)