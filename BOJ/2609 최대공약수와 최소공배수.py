import sys

def gcd(num1, num2):
    while num2 != 0:
        t = num1 % num2
        num1, num2 = num2, t
    return num1
num1, num2 = map(int, sys.stdin.readline().split())
cd = gcd(num1, num2)
print(cd)
print(num1 * num2 // cd)