def solution(cipher, code): 
    return ''.join(cipher[code-1::code])