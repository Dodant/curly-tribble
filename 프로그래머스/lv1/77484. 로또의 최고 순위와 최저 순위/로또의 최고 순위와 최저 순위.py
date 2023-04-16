def solution(lottos, win_nums):
    score = len([i for i in lottos if i in win_nums])
    zeros = lottos.count(0)
    
    if score in [0, 1]:
        if zeros == 0:
            return [6, 6]
        return [7 - score - zeros, 6]
    return [7 - score - zeros, 7 - score]
