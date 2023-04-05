# def solution(x, y):
#     answer = []
#     for i in x:
#         if i in y:
#             y = y.replace(i, '', 1)
#             answer.append(i)
#     if len(answer) == 0: return '-1'
#     return str(int(''.join(sorted(answer, reverse=True))))

# def solution(x, y):
#     x_ls = [0]*10
#     y_ls = [0]*10
#     answer = []
#     # for i in x: x_ls[int(i)] += 1
#     # for i in y: y_ls[int(i)] += 1
#     for i in '0123456789': x_ls[int(i)] = x.count(i)
#     for i in '0123456789': y_ls[int(i)] = y.count(i)
#     for a, b in zip(x_ls, y_ls): answer.append(min(a, b))
        
#     st = ''
#     for idx, itm in enumerate(answer):
#         if itm == 0: pass
#         st += str(idx)*itm

#     if st == '':
#         return '-1'
#     return str(int(st[::-1]))

# def solution(x, y):
#     answer = [0]*10
#     for i in '0123456789': 
#         answer[int(i)] = min(x.count(i), y.count(i))
        
#     st = ''
#     for idx, itm in enumerate(answer):
#         if itm == 0: pass
#         st += str(idx)*itm

#     if st == '': return '-1'
#     return str(int(st[::-1]))
from collections import Counter

def solution(x, y):
    ls = Counter(x) & Counter(y)
    answer = [0]*10
    for key, value in ls.items(): answer[int(key)] = value

    st = ''
    for idx, itm in enumerate(answer[::-1]): st += str(9-idx)*itm
    if st == '': return '-1'
    if st[0] == '0': return '0'
    return st