def solution(code):
    answer = ''
    mode = 0
    for idx, i in enumerate(code):
        if mode == 0:
            if i != '1':
                if idx % 2 == 0:
                    answer += i
            elif i == '1':
                mode = 1
        elif mode == 1:
            if i != '1':
                if idx % 2 == 1:
                    answer += i
            elif i == '1':
                mode = 0

    if answer == '':
        return 'EMPTY'
    
    return answer