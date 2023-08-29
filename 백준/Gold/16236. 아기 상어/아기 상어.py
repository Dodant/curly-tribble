def distance(space, bs_x, bs_y, tx, ty, lv):
    st = [[bs_x, bs_y, 0]]
    v = [[0] * N for _ in range(N)]

    if space[tx][ty] > lv: return -1
    
    while st:
        x, y, d = st.pop(0)
        if x == tx and y == ty: 
            return d        
        for dx, dy in ((-1,0), (0,-1), (0,1), (1,0)):
            q, w = x + dx, y + dy
            if 0 <= q < N and 0 <= w < N and v[q][w] == 0 and space[q][w] <= lv:
                st.append([q, w, d+1])
                v[q][w] = 1

import sys
input = sys.stdin.readline
N = int(input())
space = [list(map(int, input().split())) for _ in range(N)]

bs_lv = 2
bs_exp = 0
bs_x, bs_y = 0, 0

for i in range(N):
    for j in range(N):
        if space[i][j] == 9:
            bs_x, bs_y = i, j
            space[i][j] = 0
            break

answer = 0
for _ in range(N*N-1):
    edible = []
    for tx in range(N):
        for ty in range(N):
            if space[tx][ty] != 0 and space[tx][ty] < bs_lv:
                dist = distance(space, bs_x, bs_y, tx, ty, bs_lv)
                if dist == None: continue
                edible.append([tx, ty, dist])
    if len(edible) == 0: break
    edible.sort(key=lambda x: (x[2], x[0], x[1]))
    tx, ty, dist = edible.pop(0)
    space[tx][ty] = 0
    bs_x, bs_y = tx, ty
    bs_exp += 1
    if bs_exp == bs_lv:
        bs_lv += 1
        bs_exp = 0
    answer += dist
print(answer)