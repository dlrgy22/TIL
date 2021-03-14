import sys

oper = set(["+", "-"])
expression = sys.stdin.readline()[:-1]
minus_oper = False
start_idx = 0
answer = 0
for idx in range(len(expression)):
    if expression[idx] in oper:
        number = int(expression[start_idx:idx])
        if minus_oper:
            answer -= number
        else:
            answer += number
            if expression[idx] == "-":
                minus_oper = True
        start_idx = idx + 1
if minus_oper:
    answer -= int(expression[start_idx:])
else:
    answer += int(expression[start_idx:])

print(answer)