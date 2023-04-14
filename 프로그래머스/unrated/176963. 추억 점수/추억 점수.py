def solution(name, yearning, photo):
    answer = []
    mem = {}
    for k, v in zip(name, yearning):
        mem[k] = v
    for i in photo:
        score = 0
        for j in i:
            if j in mem:
                score += mem[j]
        answer.append(score)
    return answer