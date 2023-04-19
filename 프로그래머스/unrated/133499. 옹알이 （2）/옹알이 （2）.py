def solution(babbling):
    dict_ = {'aya': '1', 'ye': '2', 'woo': '3', 'ma': '4'}
    answer = 0

    for i in babbling:
        t = i
        for j in dict_.keys():
            t = t.replace(j + j, '*')
            t = t.replace(j, dict_[j])
        if t.isdigit():
            answer += 1
    return answer