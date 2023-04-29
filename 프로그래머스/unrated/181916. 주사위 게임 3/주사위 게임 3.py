from collections import Counter

def solution(a, b, c, d):
    cnt = Counter([a,b,c,d])
    ls = cnt.most_common(n=4)
    if len(ls) == 1:
        return 1111 * ls[0][0]
    elif len(ls) == 2:
        if ls[0][1] == 3:
            return (10 * ls[0][0] + ls[1][0]) ** 2
        else:
            return (ls[0][0] + ls[1][0]) * abs(ls[0][0] - ls[1][0])
    elif len(ls) == 3:
        return ls[2][0] * ls[1][0]
    else:
        return min(a,b,c,d)