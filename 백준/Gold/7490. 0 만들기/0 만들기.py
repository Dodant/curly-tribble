from itertools import product

for _ in range(int(input())):
    N = int(input())
    S = '123456789'[:N]
    O = [' '] * (N-1)
    for i in product(' +-', repeat=N-1):
        v = ''
        for q, w in zip(S, i + ('',)):
            v = v + q + w
        if eval(v.replace(' ', '')) == 0:
            print(v)
    print()