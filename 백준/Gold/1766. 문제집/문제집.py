import heapq

N, M = map(int, input().split())
JB = [list(map(int, input().split())) for _ in range(M)]
GRP = [[] for _ in range(N+1)]
JICS = [0 for _ in range(N+1)]
q = []
answer = []

for i in range(M):
    GRP[JB[i][0]].append(JB[i][1])

for i in GRP:
    if len(i) == 0:
        continue
    for j in i:
        JICS[j] += 1

for i in range(1, N+1):
    if JICS[i] == 0:
        heapq.heappush(q, i)
        
while q:
    now = heapq.heappop(q)
    if JICS[now] == 0:
        answer.append(now)
        for i in GRP[now]:
            JICS[i] -= 1
            if JICS[i] == 0:
                heapq.heappush(q, i)
    
print(' '.join(map(str, answer)))