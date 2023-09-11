N = int(input())

for i in range(N):
    H, W, N = map(int, input().split())
    if H == 1:
        XX = N
        YY = 1
    else:
        if N % H == 0:
            XX = N // H
            YY = H
        else:
            XX = N // H + 1
            if N > H: 
                YY = N % H
            else:
                YY = N
    print(f'{YY}{XX:02d}')