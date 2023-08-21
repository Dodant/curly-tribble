import sys
input = sys.stdin.readline
from collections import Counter

N = int(input())
ls = [*map(int, input().split())]
lss = [[0, 0] for _ in range(N)]
dc = dict(Counter(ls))

for i in range(len(ls)):
    lss[i][0] = ls[i]
    lss[i][1] = dc[ls[i]]

st = [0]
answer = [-1] * N

for i in range(1, N):
    while st and lss[st[-1]][1] < lss[i][1]:
        answer[st.pop()] = lss[i][0]
    st.append(i)

print(*answer)