import sys
n = int(sys.stdin.readline())
result = 3
check = [1,1,1]
for i in range(2,n + 1):
    zero = check[0] + check[1] + check[2]
    one = check[0] + check[2]
    two = check[0] + check[1]
    result = zero + one + two
    check[0] = zero
    check[1] = one
    check[2] = two
print(result % 9901)