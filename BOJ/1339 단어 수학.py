import sys

n = int(sys.stdin.readline())
word = []
count = {}
for i in range(n):
    word.append(list(sys.stdin.readline().replace('\n','')))
for word_element in word:
    c = 0
    for i in range(len(word_element) - 1, -1, -1):
        if word_element[i] in count:
            count[word_element[i]] = count[word_element[i]] + (10 ** c)
        else:
            count[word_element[i]] = (10 ** c)
        c += 1
item= sorted(count.items(), key = lambda x:x[1],reverse = True)
answer = 0
for i in range(len(item)):
    answer += item[i][1] * (9 - i)
print(answer)