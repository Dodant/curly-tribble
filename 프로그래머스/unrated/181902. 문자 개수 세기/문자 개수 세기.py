import string

def solution(my_string):
    answer = [0] * 52
    for i in my_string:
        if i.isupper():
            answer[string.ascii_uppercase.index(i)] += 1
        else:
            answer[string.ascii_lowercase.index(i) + 26] += 1
    return answer