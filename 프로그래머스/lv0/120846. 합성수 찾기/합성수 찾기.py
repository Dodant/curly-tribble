def divisor(n):
    return list(filter(lambda x: n % x == 0, list(range(1, n+1))))

def solution(n):
    return len([i for i in range(1, n+1) if len(divisor(i)) >= 3])