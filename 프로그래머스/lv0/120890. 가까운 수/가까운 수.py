def solution(array, n):
    answer = 1000
    for i in sorted(array):
        if abs(answer-n) > abs(i-n):
            answer = i
    return answer