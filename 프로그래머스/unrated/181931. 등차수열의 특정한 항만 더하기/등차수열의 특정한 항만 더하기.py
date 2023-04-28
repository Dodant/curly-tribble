def solution(a, d, included):
    answer = 0
    for i, item in enumerate(included):
        if item:
            answer += a + d * i
    return answer
        