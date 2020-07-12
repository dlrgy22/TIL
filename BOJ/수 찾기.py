import sys
n = int(sys.stdin.readline())
number = list(map(int,sys.stdin.readline().split()))
dic = {}
for i in number:
    dic[i] = 1
m = int(sys.stdin.readline())
check_num = list(map(int,sys.stdin.readline().split()))
for check in check_num:
    try:
        print(dic[check])
    except:
        print(0)
