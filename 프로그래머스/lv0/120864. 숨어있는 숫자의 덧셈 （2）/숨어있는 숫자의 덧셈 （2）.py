def solution(my_string):
    answer = []
    mem = 0
    for d, i in enumerate(my_string):
        if i.isdigit():
            mem = mem*10 + int(i)
            if d == len(my_string)-1:
                answer.append(mem)
            continue
        else:
            answer.append(mem)
            mem = 0
    return sum(answer)