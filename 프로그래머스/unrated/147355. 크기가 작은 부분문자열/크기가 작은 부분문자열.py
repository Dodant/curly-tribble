def solution(t, p):
    answer = 0
    cut = len(p)
    for i in range(len(t) - cut + 1):
        if int(p) >= int(t[i:i+cut]):
            answer += 1
    return answer