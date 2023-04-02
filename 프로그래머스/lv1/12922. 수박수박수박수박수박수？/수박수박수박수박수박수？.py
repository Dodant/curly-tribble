def solution(n):
    if n % 2 == 1: return '수박' * (n // 2) + '수'
    return '수박' * (n // 2)