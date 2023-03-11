def solution(s):
    mem = 0
    answer = 0
    for i in s.split():
        if i == 'Z':
            answer -= mem
            continue
        else:
            mem = int(i)
        answer += int(i)
    return answer