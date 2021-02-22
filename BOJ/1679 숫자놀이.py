import sys
from collections import deque

def BFS(numbers, max_use):
    count = [0 for i in range(n)]
    visit = set()
    answer = set()
    queue = deque()

    queue.appendleft([count, 0, 0])
    visit.add(tuple(count))

    while len(queue) != 0:
        use, count, num = queue.pop()
        if count > max_use:
            break
        answer.add(num)

        for i in range(n):
            next_use = use[:]
            next_use[i] += 1
            if tuple(next_use) not in visit:
                queue.appendleft([next_use, count + 1, num + numbers[i]])
                visit.add(tuple(next_use))

    return answer

def print_answer(number):
    if number % 2 == 0:
        print(f"holsoon win at {number}")
    else:
        print(f"jjaksoon win at {number}")




n = int(sys.stdin.readline())
numbers = list(map(int, sys.stdin.readline().split()))
max_use = int(sys.stdin.readline())
answer = BFS(numbers, max_use)

answer = sorted(list(answer))
for i in range(len(answer)):
    if i != answer[i]:
        print_answer(i)
        sys.exit()
print_answer(i + 1)