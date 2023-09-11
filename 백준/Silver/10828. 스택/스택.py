import sys

N = int(sys.stdin.readline())
ls = []

for _ in range(N):
    v = sys.stdin.readline().strip()
    if v == 'size':
        print(len(ls))
    elif v == 'empty':
        print(1 if len(ls) == 0 else 0)
    elif v == 'top':
        print(ls[-1] if len(ls) != 0 else -1)
    elif v == 'pop':
        print(ls.pop() if len(ls) != 0 else -1)
    else:
        ls.append(int(v.split()[1]))