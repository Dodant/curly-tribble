def solution(k, score):
    answer = []
    s = []
    for idx, i in enumerate(score):
        answer.append(i)
        answer.sort(reverse=True)
        if idx < k:
            s.append(min(answer))
        else:
            s.append(answer[k-1])
    
    return s