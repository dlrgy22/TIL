import sys

def print_answer(answer):
    for element in answer:
        print(element)

n = int(sys.stdin.readline())
stack = []
num_array = [int(sys.stdin.readline()) for i in range(n)]
answer = []

for i in range(1, n + 1):
    stack.append(i)
    answer.append("+")
    while len(num_array) != 0 and len(stack) != 0 and stack[-1] == num_array[0]:
        answer.append("-")
        del num_array[0]
        del stack[-1]

if len(num_array) == 0:
    print_answer(answer)
else:
    print("NO")