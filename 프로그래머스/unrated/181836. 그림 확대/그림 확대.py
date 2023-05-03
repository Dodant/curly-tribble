def solution(picture, k):
    answer = []
    for i in picture:
        for _ in range(k):
            answer.append(''.join([n*k for n in i]))
    return answer