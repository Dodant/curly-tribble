def solution(number, limit, power):
    answer = 0
    for i in range(1, number+1):
        if int(i ** 0.5) == i ** 0.5:
            sol = len([j for j in range(1, int(i ** 0.5)) if i % j == 0]) * 2 + 1
        else:
            sol = len([j for j in range(1, int(i ** 0.5)+1) if i % j == 0]) * 2
        if sol <= limit: answer += sol
        else: answer += power
    return answer