def solution(dartResult):
    dartResult = dartResult.replace('10', 'k')
    answer = []
    new_dart = ''
    for i in dartResult:
        if i == 'k':
            answer.append(10)
            new_dart += ' '
        elif i.isdigit():
            answer.append(int(i))
            new_dart += ' '
        else:
            new_dart += i
            
    new_dart = new_dart.split()
    for idx, item in enumerate(new_dart):
        if 'S' in item:
            answer[idx] = answer[idx] ** 1
        if 'D' in item:
            answer[idx] = answer[idx] ** 2
        if 'T' in item:
            answer[idx] = answer[idx] ** 3
        if '*' in item:
            answer[idx] = answer[idx] * 2
            if idx > 0:
                answer[idx-1] = answer[idx-1] * 2
        if '#' in item:
            answer[idx] = answer[idx] * -1
    return sum(answer)