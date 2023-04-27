def solution(n):
    if n % 2 == 1:
        return sum(list(range(1, n+1))[::2])
    if n % 2 == 0:
        return sum([i**2 for i in range(n+1) if i % 2 == 0])