def solution(s):
    ls = list(map(int, s.split()))
    return f"{min(ls)} {max(ls)}"