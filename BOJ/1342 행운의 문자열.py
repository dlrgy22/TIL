import sys

def next_word(word):
    global length, s
    if sum(visit) == length:
        result.add(s)
        return

    for i in range(length):
        if not visit[i]:
            if word != string[i]:
                s += string[i]
                visit[i] = True
                word = string[i]
                next_word(word)
                visit[i] = False
                s = s[:-1]
                word = s[-1]

string = list(sys.stdin.readline().replace('\n', ''))
result = set()
s = ''
length = len(string)
visit = [False for i in range(length)]
for i in range(length):
    word = string[i]
    s += word
    visit[i] = True
    next_word(word)
    visit[i] = False
    s = ''

if string == '':
    print(1)
    sys.exit()

print(len(result))

