def solution(my_str, n):
    answer = []
    b = ''
    cnt = 0
    for d, i in enumerate(my_str):
        if len(b) == n-1:
            b = b + i
            answer.append(b)
            cnt = 0
            b = ''
        else:
            if d == len(my_str)-1:
                b = b + i
                answer.append(b)
            b = b + i
            cnt += 1
    return answer