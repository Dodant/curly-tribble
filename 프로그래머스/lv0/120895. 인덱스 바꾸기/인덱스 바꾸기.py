def solution(my, n1, n2):
    my = list(my)
    my[n1], my[n2] = my[n2], my[n1]
    return ''.join(my)