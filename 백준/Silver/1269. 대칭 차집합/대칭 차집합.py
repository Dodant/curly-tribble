N, M = map(int, input().split())
ls = set(list(input().split()))
ms = set(list(input().split()))
print(len(ls - ms) + len(ms - ls))