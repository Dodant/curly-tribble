def solution(score):
    answer = []
    avg = [(a+b)+10000 for a, b in score]
    high = sorted(avg, reverse=True)
    
    rank = 0
    cnt = 0
    prev = 0
    for idx, item in enumerate(high):
        if prev != avg[avg.index(item)]:
            rank = rank + cnt + 1
            cnt = 0
        else:
            cnt += 1
        prev = avg[avg.index(item)]
        avg[avg.index(item)] = rank
        
    return avg