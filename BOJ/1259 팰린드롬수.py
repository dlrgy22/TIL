import sys

while True:
    num = sys.stdin.readline()[:-1]
    if num == "0":
        break
    print("yes") if num == num[::-1] else print("no")