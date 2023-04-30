def solution(my_strings, parts):
    answer = ''
    for st, i in zip(my_strings, parts):
        answer += st[i[0]: i[1]+1]
    return answer