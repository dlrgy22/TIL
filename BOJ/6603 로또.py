import sys
import itertools

number = []
while True:
    num = list(map(int,sys.stdin.readline().split()))
    if num[0] == 0:
        break
    else:
        number.append(num[1:])
for i in number:
    lotto = itertools.combinations(i,6)
    for j in lotto:
        for k in j:
            print(k,end = " ")
        print()
    print()