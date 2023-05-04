def solution(rank, attendance):
    answer = []
    for i, r, a in zip(range(0, len(rank)), rank, attendance):
        if a:
            answer.append([i, r])
    answer.sort(key=lambda x: x[1])
    return answer[0][0] * 10000 + answer[1][0] * 100 + answer[2][0]