def solution(s):
    s = s.title()
    for idx, item in enumerate(s):
        if item.isdigit():
            s = s[:idx+1] + s[idx+1].lower() + s[idx+2:] 
    return s