def check(i):
    if i % 3 == 0 or '3' in str(i): 
        return check(i + 1)
    else:
        return i

def solution(n):
    i = 0
    for _ in range(n):
        i += 1
        i = check(i)
    return i