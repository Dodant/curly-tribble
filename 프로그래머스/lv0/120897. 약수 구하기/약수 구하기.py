def solution(n):
    return list(filter(lambda x: n % x == 0, list(range(1, n+1))))