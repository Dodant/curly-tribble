def solution(k, tangerine):
    answer = 0
    dt = {}
    
    for i in range(len(tangerine)):
        if tangerine[i] not in dt:
            dt[tangerine[i]] = 1
        else:
            dt[tangerine[i]] += 1
            
    values = sorted(list(dt.values()), reverse=True)
    
    for i in range(len(values)):
        if k > values[i]:
            k -= values[i]
            answer += 1
        elif k <= values[i]:
            answer += 1
            break
        else:
            break

    return answer