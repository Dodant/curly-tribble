def solution(id_list, report, k):
    answer = []
    report = list(set(report))
    report2 = [tuple(report[i].split()) for i in range(len(report))]
    
    dt = {}

    for i in report2:
        rpted_user = i[1]
        if rpted_user not in dt:
            dt[rpted_user] = 1
        else: 
            dt[rpted_user] += 1
    
    block = [i for i in dt if dt[i] >= k]
    
    dtdt = {}
    for i in id_list:
        dtdt[i] = set()
    
    for i in report2:
        user, rpted_user = i[0], i[1]
        if rpted_user in block :
            dtdt[user].add(rpted_user)
            
    for i in dtdt:
        answer.append(len(dtdt[i]))
    return answer