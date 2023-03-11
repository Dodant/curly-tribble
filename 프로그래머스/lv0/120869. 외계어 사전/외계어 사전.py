def solution(spell, dic):
    answer = []
    for i in dic:
        s = i
        for j in spell:
            s = s.replace(j, '0', 1)
        answer.append(s)
    if "0"*len(spell) in answer: return 1
    return 2