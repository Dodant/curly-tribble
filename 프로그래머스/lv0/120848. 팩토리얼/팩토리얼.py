def facto(n):
    if n == 1:
        return 1
    return n * facto(n-1)

def solution(n):
    for i in range(10,0,-1):
        if n >= facto(i):
            return i