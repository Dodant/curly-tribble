def solution(s):
    stck = list(s[0:2])
    for i in s[2:]:
        if len(stck) >= 2 and stck[-2] == stck[-1]:
            stck.pop()
            stck.pop()
        stck.append(i)
    if len(stck) >= 2 and stck[-2] == stck[-1]:
            stck.pop()
            stck.pop()
    if len(stck) == 0:
        return 1
    return 0