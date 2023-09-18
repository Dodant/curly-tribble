n = int(input())
for _ in range(n):
    st = input().split()
    print(' '.join(list(map(lambda x:x[::-1], st))))