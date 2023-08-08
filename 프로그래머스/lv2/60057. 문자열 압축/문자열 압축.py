def solution(s):
    
    answers = []
    
    if len(s) == 1:
        return 1
    
    for window_len in range(1, len(s) // 2 + 1):
        cnt = 1
        answer = ""
        first_s = split_string(s, window_len)
        second_s = split_string(s, window_len)[1:] + [""]
        
        for i, j in zip(first_s, second_s):
            if i == j:
                cnt += 1
            else:
                if cnt == 1:
                    answer = answer + i
                else:
                    answer = answer + str(cnt) + i
                    cnt = 1

        answers.append(answer)
    return min([len(i) for i in answers])
        


def split_string(s, length):
    bag = []
    for i in range(0, len(s), length):
        bag.append(s[i:i+length])
    return bag