import sys

N = int(sys.stdin.readline().rstrip())
ls = []
for _ in range(N):
    v = sys.stdin.readline().rstrip()
    if v == '2':
        print(ls.pop() if len(ls) != 0 else -1)
    elif v == '3':
        print(len(ls))
    elif v == '4':
        print(1 if len(ls) == 0 else 0)
    elif v == '5':
        print(ls[-1] if len(ls) != 0 else -1)
    else:
        ls.append(int(v.split()[1]))