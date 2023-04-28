def solution(numLog):
    answer = ''
    first = numLog[0]
    for i in numLog:
        if i == first + 1: 
            first += 1
            answer += 'w'
        elif i == first - 1: 
            first -= 1
            answer += 's'
        if i == first + 10: 
            first += 10
            answer += 'd'
        if i == first - 10: 
            first -= 10
            answer += 'a'
    return answer