def solution(s):
    answer = []
    alphabet = 'abcdefghijklmnopqrstuvwxyz'
    alphabet_list = [-2] * len(alphabet)
    for idx, letter in enumerate(s):
        if alphabet_list[alphabet.index(letter)] == -2:
            answer.append(-1)
            alphabet_list[alphabet.index(letter)] = idx
        else:
            answer.append(idx - alphabet_list[alphabet.index(letter)])
            alphabet_list[alphabet.index(letter)] = idx
    
    return answer