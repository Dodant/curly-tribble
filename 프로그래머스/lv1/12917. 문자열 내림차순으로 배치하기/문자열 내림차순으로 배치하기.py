def solution(s):
    u, l = [], []
    for i in s:
        if i.isupper(): u.append(i)
        if i.islower(): l.append(i)
    return ''.join(sorted(l))[::-1] + ''.join(sorted(u))[::-1]