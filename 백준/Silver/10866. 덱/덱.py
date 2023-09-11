import sys

N = int(sys.stdin.readline())
ls = []

for _ in range(N):
    v = sys.stdin.readline().strip()
    if v == 'size':
        print(len(ls))
    elif v == 'empty':
        print(1 if len(ls) == 0 else 0)
    elif v == 'front':
        print(ls[0] if len(ls) != 0 else -1)
    elif v == 'back':
        print(ls[-1] if len(ls) != 0 else -1)
    elif v == 'pop_front':
        print(ls.pop(0) if len(ls) != 0 else -1)
    elif v == 'pop_back':
        print(ls.pop() if len(ls) != 0 else -1)
    elif v.split()[0] == 'push_front':
        ls = [int(v.split()[1])] + ls
    elif v.split()[0] == 'push_back':
        ls.append(int(v.split()[1]))