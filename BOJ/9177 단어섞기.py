import sys

def check_char(string1, string2,  string3):
    check_char= set(list(string1) +  list(string2))
    for idx in range(len(string3)):
        if string3[idx] not in check_char:
            return False
    return True

def check_point(str1_point, str2_point, str3_point):

    global check
    if check:
        return
    if str1_point < len(string1) and string1[str1_point] == string3[str3_point]:
        if str3_point == len(string3) - 1:
            check = True
            return
        check_point(str1_point + 1, str2_point, str3_point + 1)
    if str2_point < len(string2) and string2[str2_point] == string3[str3_point]:
        if str3_point == len(string3) - 1:
            check = True
            return
        check_point(str1_point, str2_point + 1, str3_point + 1)

n = int(sys.stdin.readline())
for i in range(n):
    string1, string2, string3 = sys.stdin.readline().split()
    if not check_char(string1, string2, string3):
        print("Data set {}: no".format(i + 1))
        continue

    check = False
    str1_point = 0
    str2_point = 0
    str3_point = 0
    check_point(str1_point, str2_point, str3_point)
    if check:
        print("Data set {}: yes".format(i + 1))
    else:
        print("Data set {}: no".format(i + 1))