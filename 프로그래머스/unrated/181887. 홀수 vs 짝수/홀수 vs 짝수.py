def solution(num_list):
    answer_o = 0
    answer_e = 0
    for idx, item in enumerate(num_list):
        if idx % 2 == 0:
            answer_o += item
        else:
            answer_e += item
    return max(answer_o, answer_e)