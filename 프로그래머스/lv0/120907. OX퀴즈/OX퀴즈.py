def solution(quiz):
    answer = []
    for i in quiz:
        s = i.split('=')
        if eval(s[0]) == int(s[1].strip()):
            answer.append('O')
        else:
            answer.append('X')
    return answer