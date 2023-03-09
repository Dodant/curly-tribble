def solution(poly):
    x_, p_ = 0, 0
    if len(poly) in [1,2]: return poly
    for i in poly.split('+'):
        j = i.strip()
        if 'x' in j:
            if j == 'x': x_ += 1
            else: x_ += int(j[:-1])
            continue
        else: p_ += int(j)
    if x_ != 0 and p_ == 0: return f'{x_}x'
    if x_ == 0 and p_ != 0: return str(p_)
    if x_ == 0 and p_ == 0: return '0'
    if x_ == 1: return f'x + {p_}'
    return f'{x_}x + {p_}'
    