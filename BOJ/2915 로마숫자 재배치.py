import itertools
import math
import sys

def roma_to_arbic(roma):
    for i in range(99):
        if roma == number[i]:
            return i + 1
    return 100

number = ['I', 'II', 'III', 'IV', 'V', 'VI', 'VII', 'VIII', 'IX']
t_digit = ['X', 'XX', 'XXX', 'XL', 'L', 'LX', 'LXX', 'LXXX', 'XC']
for i in range(9):
    for j in range(10):
        if j == 0:
            number.append(t_digit[i])
        else:
            number.append(t_digit[i] + number[j - 1])


roma_number = list(sys.stdin.readline().replace('\n',''))
small_num = math.inf
for per in itertools.permutations(roma_number,len(roma_number)):
    check_num = ""
    for per_element in per:
        check_num += per_element
    arbic = roma_to_arbic(check_num)
    small_num = min(arbic, small_num)

print(number[small_num - 1])