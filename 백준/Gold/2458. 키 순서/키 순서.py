import sys

input = sys.stdin.readline
n, m = map(int, input().split())

map_ = [[1000] * n for _ in range(n)]
for _ in range(m):
    a, b = map(int, input().split())
    map_[a-1][b-1] = 1


for k in range(n):
    for i in range(n):
        for j in range(n):
            if map_[i][j] > map_[i][k] + map_[k][j]:
                map_[i][j] = 1

answer = 0
for i in range(n):
    ax = len([q for q in map_[i] if q == 1])
    for j in range(n):
        if map_[j][i] == 1: 
            ax += 1
    if ax == n-1: 
        answer += 1
print(answer)