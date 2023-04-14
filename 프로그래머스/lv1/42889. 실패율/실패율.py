def solution(N, stages):
    stage = dict()
    answer = []
    for i in range(1, N+1): stage[i] = 0
    
    for i in range(1, N+1):
        leng = len([j for j in stages if j >= i])
        if 0 == leng:
            stage[i] = 0
        else:
            stage[i] = len([j for j in stages if j == i]) / leng

    for i in sorted(stage, reverse=True, key=lambda x: stage[x]):
        answer.append(i)
    return answer