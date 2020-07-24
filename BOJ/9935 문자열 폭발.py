import sys

def check_explosion(result):
    if result[len(result) - len(explosion):] == explosion:
        return True

def explosion_string():
    result = []
    for string_element in string:
        result.append(string_element)
        if len(result) >= len(explosion):
            if check_explosion(result):
                for i in range(len(explosion)):
                    result.pop()

    return result


string = list(sys.stdin.readline().replace('\n',''))
explosion = list(sys.stdin.readline().replace('\n',''))
result = explosion_string()
if len(result) == 0:
    print('FRULA')
else:
    for i in result:
        print(i,end = '')
