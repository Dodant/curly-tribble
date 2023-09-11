import sys
from collections import deque

N = int(sys.stdin.readline().rstrip())
ls = deque()

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
    elif v == 'pop':
        print(ls.popleft() if len(ls) != 0 else -1)
    else:
        ls.append(int(v.split()[1]))