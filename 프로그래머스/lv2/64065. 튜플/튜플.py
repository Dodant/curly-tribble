def solution(s):
    s = s[2:-2].split('},{')
    answer = []
    dict_ = {}
    for i in s:
        answer.append(list(map(int, i.split(','))))
        
    for i in answer: 
        for j in i: dict_[j] = 0
    for i in answer: 
        for j in i: dict_[j] += 1

    dict_ = sorted(dict_, key=lambda x: dict_[x], reverse=True)
    return dict_