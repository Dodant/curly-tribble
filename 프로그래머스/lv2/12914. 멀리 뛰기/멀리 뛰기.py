ls = [0, 1, 2, 3]

def solution(n):
    if n < 4: 
        return n
    else:
        for i in range(4, n+1):
            ls.append(ls[i-1] + ls[i-2])
        return ls[n] % 1234567