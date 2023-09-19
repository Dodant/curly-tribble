T = int(input())
bld = [[0 for _ in range(14+1)] for _ in range(14+1)]
for i in range(0, 14+1):
    bld[0][i] = i + 1
for i in range(1, 14+1):
    bld[i][0]= 1
    
for i in range(1, 14+1):
    for j in range(1, 14+1):
        bld[i][j] = bld[i-1][j] + bld[i][j-1]
        
for _ in range(T):
    k = int(input())
    n = int(input())
    print(bld[k][n-1])