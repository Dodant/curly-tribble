def solution(n):
    answer = []
    while True:
        answer.append(str(n % 3))
        n //= 3
        if n == 0:
            break
    return int(''.join(answer), 3)
