def solution(arr1, arr2):

    n, m = len(arr1), len(arr1[0])
    m, k = len(arr2), len(arr2[0])

    base = [[0] * k for _ in range(n)]
    
    for i in range(n):
        for j in range(k):
            base[i][j] = sum(arr1[i][l] * arr2[l][j] for l in range(m))
    
    return base
