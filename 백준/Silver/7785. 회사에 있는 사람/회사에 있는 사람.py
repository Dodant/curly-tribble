N = int(input())
ls = set()
for _ in range(N):
    a, b = input().split()
    if b == 'enter':
        ls.add(a)
    elif b == 'leave':
        ls.remove(a)
print('\n'.join(sorted(ls, reverse=True)))