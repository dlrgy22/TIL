import sys
def find_deciaml():
    decimal = [True for i in range(1001)]
    decimal[0], decimal[1] = False, False
    for i in range(2, 1001):
        if decimal[i] == True:
            for j in range(2*i, 1001, i):
                decimal[j] = False
    return decimal

n = int(sys.stdin.readline())
num_list = list(map(int, sys.stdin.readline().split()))
decimal = find_deciaml()
count = 0
for num in num_list:
    if decimal[num]:
        count += 1
print(count)
