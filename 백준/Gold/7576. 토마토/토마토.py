from collections import deque
import sys

input = sys.stdin.readline
n, m = map(int, input().split())
arr = [list(map(int, input().split())) for _ in range(m)]

queue = deque()
for i in range(m):
    for j in range(n):
        if arr[i][j] == 1:
            queue.append([i, j])
dx = [0, 0, -1, 1]
dy = [-1, 1, 0, 0]

while queue:
    i, j = queue.popleft()
    for n_ in range(4):
        z, x = i + dx[n_], j + dy[n_]
        if 0 <= z < m and 0 <= x < n and arr[z][x] == 0:
            arr[z][x] = arr[i][j] + 1
            queue.append([z, x])
ans = 0
for line in arr:
    for tomato in line:
        if tomato == 0:
            print(-1)
            exit()
    ans = max(ans, max(line))
print(ans-1)