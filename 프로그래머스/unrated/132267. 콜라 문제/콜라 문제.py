def solution(a, b, n):
    answer = 0
    while n >= 2:
        div, mod = divmod(n, a)
        answer += div * b
        n = div * b + mod
        if n < a:
            break
        
    return answer