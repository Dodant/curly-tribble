N = int(input())
ls = ['' for _ in range(N)]
for i in range(N):
    ls[i] = list(map(int, input().split()))
ls.sort(key=lambda x: (x[1], x[0]))
print('\n'.join([' '.join(map(str, i)) for i in ls]))