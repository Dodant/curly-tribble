def solution(s, skip, index):
    alphabet = "abcdefghijklmnopqrstuvwxyz"
    alphabet = [x for x in alphabet if x not in list(set(skip))]
    alphabet = alphabet * 5
    
    answer = []
    for i in s:
        answer.append(alphabet[alphabet.index(i)+index])
    
    return ''.join(answer)