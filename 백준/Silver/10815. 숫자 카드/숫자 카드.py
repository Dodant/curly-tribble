N = int(input())
ls = set(map(int, input().split()))
M = int(input().strip())
fs = list(map(int, input().split()))

answer = [0 for _ in range(M)]
for i in range(len(fs)):
    if fs[i] in ls:
        answer[i] = 1
print(*answer)