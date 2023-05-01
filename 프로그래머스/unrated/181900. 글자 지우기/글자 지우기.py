def solution(my_string, indices):
    my_string = [i for i in my_string]
    for i in indices:
        my_string[i] = ' '
    return ''.join(my_string).replace(' ', '')