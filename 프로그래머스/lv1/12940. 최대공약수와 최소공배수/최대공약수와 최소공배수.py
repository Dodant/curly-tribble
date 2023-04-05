def divisor(n):
    return [i for i in range(1, n+1) if n % i == 0]

def gcd(n,m):
    n_ls, m_ls = divisor(n), divisor(m)
    return max([i for i in n_ls if i in m_ls])
    
def lcm(n,m):
    return n * m // gcd(n,m)
    
def solution(n, m):
    return [gcd(n,m), lcm(n,m)]