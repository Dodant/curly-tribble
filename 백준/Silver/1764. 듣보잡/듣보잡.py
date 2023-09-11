N, M = map(int, input().split())
ls = set([input() for i in range(N)])
ms = set([input() for i in range(M)])
answer = list(ls & ms)
answer.sort()
print(len(answer))
for i in answer:
    print(i)