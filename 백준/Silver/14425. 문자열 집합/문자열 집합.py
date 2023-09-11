N, M = map(int, input().split())
ls = set()
answer = 0
for _ in range(N):
    ls.add(input())
for _ in range(M):
    if input() in ls:
        answer += 1
print(answer)