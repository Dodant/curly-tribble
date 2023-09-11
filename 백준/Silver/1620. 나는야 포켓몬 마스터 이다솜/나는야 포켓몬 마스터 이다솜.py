import sys

N, M = map(int, sys.stdin.readline().split())
dct1 = {}
for i in range(N):
    g = sys.stdin.readline().strip()
    dct1[g] = i+1
    dct1[str(i+1)] = g
for _ in range(M):
    m = sys.stdin.readline().strip()
    print(dct1[m])