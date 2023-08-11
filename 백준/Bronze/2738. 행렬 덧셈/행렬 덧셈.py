n, m = map(int, input().split())

a = [list(map(int, input().split())) for i in range(n)]
b = [list(map(int, input().split())) for i in range(n)]

for i in range(n):
    for x, y in zip(a[i], b[i]):
        print(x + y, end=" ")
    print()