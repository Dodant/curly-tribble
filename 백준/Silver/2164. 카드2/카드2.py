from collections import deque

N = int(input())
ls = deque([i for i in range(1, N+1)])
while len(ls) > 1:
    ls.popleft()
    ls.append(ls.popleft())
print(ls[0])