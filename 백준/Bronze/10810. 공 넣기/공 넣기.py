N, M = map(int, input().split())
arr = [0 for i in range(1, N+1)]
for _ in range(M):
    a, b, c = map(int, input().split())
    for i in range(a-1, b):
        arr[i] = c
print(*arr)