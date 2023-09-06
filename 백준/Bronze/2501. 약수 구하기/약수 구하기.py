N, K = map(int, input().split())
ls = [i for i in range(1, N+1) if N % i == 0]
print(0 if len(ls) < K else ls[K-1])