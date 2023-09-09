def find(data, p, step):
    if step == 6: return
    if p != '': data.append(p)
    for c in ['A', 'E', 'I', 'O', 'U']:
        find(data, ''.join([p,c]), step+1)
        
def solution(word):
    data = []
    find(data, '', 0)
    return data.index(word) + 1