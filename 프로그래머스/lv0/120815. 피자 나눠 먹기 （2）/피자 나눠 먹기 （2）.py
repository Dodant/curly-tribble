def divisor(n):
    return filter(lambda x: n % x == 0, list(range(1, n+1)))

def GCD(n, m):
    return max(set(divisor(n)) & set(divisor(m)))

def LCM(n, m):
    return n * m // GCD(n,m)

def solution(n):
    return LCM(n, 6) // 6