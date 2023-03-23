def solution(my_string):
    ls = []
    for i in my_string:
        if i not in ['a','u','i','o','e']:
            ls.append(i)
    return ''.join(ls)