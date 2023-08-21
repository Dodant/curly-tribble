import sys
input = sys.stdin.readline

N = int(input())
ls = list(map(int, input().split()))

st = [0]
answer = [0] * N

for i in range(1, N):
    while st and ls[st[-1]] < ls[i]:
        answer[st.pop()] = ls[i]
    st.append(i)

while st:
    answer[st.pop()] = -1

print(*answer)