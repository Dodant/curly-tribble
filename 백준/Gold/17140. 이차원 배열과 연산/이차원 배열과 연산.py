r, c, k = map(int, input().split())
r, c = r-1, c-1

A = [list(map(int, input().split())) for _ in range(3)]

time = 0

def count_dict(L):
    d = {}
    for i in L:
        if i == 0:
            continue
        if i not in d:
            d[i] = 1
        else:
            d[i] += 1
            
    f = []
    for k,v in d.items():
        f.append((k, v))
    f.sort(key=lambda x: (x[1], x[0]))
    
    g = []
    for k, v in f:
        g += [k, v]
        
    return g, len(g)

def transpose(A):
    row, col = len(A), len(A[0])
    
    B = []
    for i in range(col):
        B.append([])
        for j in range(row):
            B[i].append(A[j][i])
    return B

def count_matrix(A):
    row, col = len(A), len(A[0])
    
    max_length = 0
    B = []
    if row >= col:
        for i in range(row):
            L, length = count_dict(A[i])
            
            if max_length == 100: max_length = 100
            elif length > max_length: max_length = length
            if len(L) > 100: L = L[:100]
            B.append(L)
            
        for i in range(len(B)):
            for _ in range(max_length - len(B[i])):
                B[i].append(0)
    else:
        A = transpose(A)
        for i in range(col):
            L, length = count_dict(A[i])

            if max_length == 100: max_length = 100
            elif length > max_length: max_length = length
            if len(L) > 100: L = L[:100]
            B.append(L)
            
        for i in range(len(B)):
            for _ in range(max_length - len(B[i])):
                B[i].append(0)
        B = transpose(B)
    return B
    
while True:
    try:
        if A[r][c] == k:
            print(time)
            break
    except IndexError:
        pass
    
    if time >= 100:
        print(-1)
        break
    
    A = count_matrix(A)
    time += 1
