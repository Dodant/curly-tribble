def solution(n, lost, reserve):
    lost_ = list(set(lost) - set(reserve))
    reserve_ = list(set(reserve) - set(lost))
    lost_r = sorted(lost_)
    lost_l = sorted(lost_)
    lost_r_ = sorted(lost_r)
    lost_l_ = sorted(lost_l)
    reserve_ = sorted(reserve_)

    for i in lost_r_:
        if i + 1 in reserve_:
            if i - 1 in reserve_:
                reserve_.remove(i-1)
                lost_r.remove(i)
            else:
                reserve_.remove(i+1)
                lost_r.remove(i)
        elif i - 1 in reserve_:
            reserve_.remove(i-1)
            lost_r.remove(i)

    for i in lost_l_:
        if i - 1 in reserve_:
            reserve_.remove(i-1)
            lost_l.remove(i)
        elif i + 1 in reserve_:
            reserve_.remove(i+1)
            lost_l.remove(i)

    return n - min(len(lost_r), len(lost_l))