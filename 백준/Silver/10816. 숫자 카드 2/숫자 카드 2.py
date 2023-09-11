N = int(input())
ls = list(input().split())
M = int(input())
ms = list(input().split())
answer = []
dct = {}
for i in ls:
    if i not in dct:
        dct[i] = 1
    else:
        dct[i] += 1
for m in ms:
    if m not in dct:
        answer.append(0)
    else:
        answer.append(dct[m])
print(*answer)