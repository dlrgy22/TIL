book,carry = map(int, input().split(' '))
left =[]
right = []
work = map(int,input().split(' '))
for i in work:
    if i > 0:
        right.append(i)
    else:
        left.append(abs(i))
if len(left) ==0:
    left.append(0)
if len(right) ==0:
    right.append(0)
right.sort(reverse=True)
left.sort(reverse=True)
i=0
res=0
while(i<len(right)):
    res += right[i]*2
    i +=carry
i=0
while(i<len(left)):
    res += left[i]*2
    i += carry
res -= max(left[0],right[0])
print(res)