import sys
import math

def check_year(lcm ,num_1, num_2, m, n):
    while lcm >= num_1 and lcm >= num_2:
        if num_1 == num_2:
            return num_1
        elif num_1 > num_2:
            num_2 += n
        else:
            num_1 += m
    return -1

t = int(sys.stdin.readline())
for i in range(t):
    m, n, x, y = map(int,sys.stdin.readline().split())
    lcm = (m * n) / math.gcd(m, n)
    print(check_year(lcm, x, y, m, n))
