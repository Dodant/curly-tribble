def solution(myString, pat):
    my_s = ''.join(['B' if i == 'A' else 'A' for i in myString])
    return 1 if pat in my_s else 0