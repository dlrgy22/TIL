import sys
n = int(sys.stdin.readline())
word_set = set()
for i in range(n):
    word_set.add(sys.stdin.readline()[:-1])
sorted_word = sorted(word_set, key=lambda x: (len(x), x))
for word in sorted_word:
    print(word)
