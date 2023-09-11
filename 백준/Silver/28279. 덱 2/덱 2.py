import sys
from collections import deque

N = int(sys.stdin.readline().rstrip())
ls = deque()

for _ in range(N):
    v = sys.stdin.readline().strip()
    if v == '3':
        print(ls.popleft() if len(ls) != 0 else -1)
    elif v == '4':
        print(ls.pop() if len(ls) != 0 else -1)
    elif v == '5':
        print(len(ls))
    elif v == '6':
        print(1 if len(ls) == 0 else 0)
    elif v == '7':
        print(ls[0] if len(ls) != 0 else -1)
    elif v == '8':
        print(ls[-1] if len(ls) != 0 else -1)
    elif v.split()[0] == '1':
        ls.appendleft(int(v.split()[1]))
    else:
        ls.append(int(v.split()[1]))