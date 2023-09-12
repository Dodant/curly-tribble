A, B = map(int, input().strip().split())
K, X = map(int, input().strip().split())

ls = set([i for i in range(A, B+1)])
ms = set([i for i in range(K-X, K+X+1)])
print('IMPOSSIBLE' if len(ls.intersection(ms)) == 0 else len(ls.intersection(ms)))