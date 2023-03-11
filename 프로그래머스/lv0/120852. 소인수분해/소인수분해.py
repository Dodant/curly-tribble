def divisor(n):
    return list(filter(lambda x: n % x == 0, list(range(1, n+1))))

def solution(n):
    div = divisor(n)
    return [i for i in div if len(divisor(i)) == 2]
    