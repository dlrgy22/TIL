def AND(x1,x2):
    w1, w2, theta = 0.5, 0.5, 0.7
    tmp = x1 * w1 + x2 * w2
    if tmp <= theta:
        return 0
    else:
        return 1

x1,x2 = map(int,input().split())
if (x1 == 0 or x1 ==  1) and (x2 == 0 or x2 == 1):
    print(AND(x1,x2))
