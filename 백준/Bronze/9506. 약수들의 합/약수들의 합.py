while True:
    N = int(input())
    if N == -1: break
    ls = [i for i in range(1, N) if N % i == 0]
    if sum(ls) == N:
        print(f'{N} = {" + ".join([*map(str, ls)])}')
    else:
        print(f'{N} is NOT perfect.')