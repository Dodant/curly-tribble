def solution(i):
    answer = []
    i = i.replace('a', ' ')
    i = i.replace('b', ' ')
    i = i.replace('c', ' ')
    if i.strip() == '':
        return ["EMPTY"]
    return i.split()