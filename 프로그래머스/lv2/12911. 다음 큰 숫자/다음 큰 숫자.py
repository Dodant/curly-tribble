def solution(n):
    answer = n
    while True:
        answer += 1
        if bin(n).replace('0b','').count('1') == bin(answer).replace('0b','').count('1'):
            return answer
        
        