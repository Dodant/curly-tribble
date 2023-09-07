M = int(input())
N = int(input())
ls = [i for i in range(M, N+1) if len([j for j in range(1, i+1) if i % j == 0]) == 2]
if len(ls) == 0: 
    print(-1)
else:
    print(sum(ls))
    print(min(ls))