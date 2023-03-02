def solution(n):
    answer = []
    while True:
        answer.append(str(n % 3))
        n //= 3
        if n == 0:
            break
    result = 0
    for idx, item in enumerate(''.join(answer)[::-1]):
        result += int(item) * 3 ** idx
    return result