def divisor(n):
    return filter(lambda x: n % x == 0, list(range(1, n+1)))

def GCD(n, m):
    return max(set(divisor(n)) & set(divisor(m)))

def solution(numer1, denom1, numer2, denom2):
    temp_num = numer1 * denom2 + numer2 * denom1
    temp_den = denom1 * denom2
    return [temp_num // GCD(temp_num, temp_den), temp_den // GCD(temp_num, temp_den)]     