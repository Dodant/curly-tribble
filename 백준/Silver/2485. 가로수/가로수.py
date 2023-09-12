import math

N = int(input())
ls = []
prev = int(input())
for i in range(N-1):
    cur = int(input())
    ls.append(cur-prev)
    prev = cur

minnum = math.gcd(*ls)
answer = 0
for i in ls:
    if i == minnum: 
        continue
    else:
        answer += i//minnum - 1
print(answer)