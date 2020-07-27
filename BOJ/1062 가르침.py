import sys
import itertools

n, k = map(int, sys.stdin.readline().split())

Essential = ['a', 'n', 't', 'i', 'c']
word = [set() for i in range(n)]
check_alpa = set()

for i in range(n):
    w = sys.stdin.readline().replace('\n', '')
    for j in range(4, len(w) - 4):
        if w[j] not in Essential:
            word[i].add(w[j])
            check_alpa.add(w[j])
result = 0

if k < 5:
    print(0)
    sys.exit()

for per_element in list(itertools.combinations(check_alpa, k - 5)):
    count = 0
    for word_element in word:
        check = True
        for check_word in word_element:
            if check_word not in per_element:
                check = False
                break
        if check:
            count += 1
    if result < count :
        result = count

if result == 0 and len(check_alpa) <= k - 5:
    result = n


print(result)





