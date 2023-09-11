N = int(input())
ls = set(list(map(int, input().split())))
M = int(input())
ms = list(map(int, input().split()))
for m in ms:
    print(1 if m in ls else 0)