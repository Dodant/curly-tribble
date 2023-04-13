def solution(answers):
    fir_person = [1, 2, 3, 4, 5]
    sec_person = [2, 1, 2, 3, 2, 4, 2, 5]
    thr_person = [3, 3, 1, 1, 2, 2, 4, 4, 5, 5]
    score = dict()
    
    for i in range(1, 4):
        score[str(i)] = 0
        
    for idx, item in enumerate(answers):
        if item == fir_person[idx%len(fir_person)]: score['1'] += 1
        if item == sec_person[idx%len(sec_person)]: score['2'] += 1
        if item == thr_person[idx%len(thr_person)]: score['3'] += 1
    
    return sorted([int(k) for k, v in score.items() if v == max(score.values())])
    