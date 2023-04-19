def solution(s):
    answer = 0
    same_len = 0
    diff_len = 0
    for i in s:
        if same_len == diff_len == 0:
            x = i
        if x != i:
            diff_len += 1
        elif x == i:
            same_len += 1
        if diff_len == same_len:
            answer += 1
            same_len = 0
            diff_len = 0
    if same_len != diff_len:
        answer += 1
    return answer