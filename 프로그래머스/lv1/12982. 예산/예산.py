def solution(d, budget):
    answer = 0
    summ = 0
    for i in sorted(d):
        summ += i
        if summ > budget: break
        answer += 1
    return answer