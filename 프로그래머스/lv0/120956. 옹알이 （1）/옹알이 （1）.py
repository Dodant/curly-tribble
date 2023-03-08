def solution(babbling):
    answer = []
    for i in babbling:
        if 'aya' in i: i = i.replace('aya', '#', 1)
        if 'ye' in i: i = i.replace('ye', '#', 1)
        if 'woo' in i: i = i.replace('woo', '#', 1)
        if 'ma' in i: i = i.replace('ma', '#', 1)
        answer.append(i)
    cnt = 0
    for i in answer:
        if i in ['#', '##', '###', '####']:
            cnt += 1
    return cnt