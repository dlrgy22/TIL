import sys
while True:
    sides = sorted(list(map(int, sys.stdin.readline().split())))
    if sides == [0, 0, 0]:
        break
    print("right") if sides[2]**2 == (sides[0]**2 + sides[1]**2) else print("wrong")
