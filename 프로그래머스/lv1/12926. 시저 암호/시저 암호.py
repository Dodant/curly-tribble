def solution(s, n):
    answer = []
    for i in s:
        if i == ' ':
            answer.append(i)
        elif i.isupper():
            answer.append(chr(((ord(i) + n) - 65) % 26 + 65))
        elif i.islower():
            answer.append(chr(((ord(i) + n) - 97) % 26 + 97))
    return ''.join(answer)