def solution(today, terms, privacies):
    
    dic = {}
    answer = []
    for i in terms:
        term, month = i.split()
        dic[term] = int(month)

    for i in range(len(privacies)):
        date, term = privacies[i].split()
        yyyy, mm, dd = map(int, date.split('.'))
        tyyyy, tmm, tdd = map(int, today.split('.'))
        if tyyyy*12*28 + tmm*28 + tdd > yyyy*12*28 + mm*28 + dd + dic[term]*28 - 1:
            answer.append(i+1)
        
#         mm += dic[term]
#         yyyy += mm // 12
#         mm = mm % 12
#         dd -= 1
        
#         if dd <= 0:
#             dd = 28
#             mm -= 1
            
#         if mm <= 0:
#             mm = 12
#             yyyy -= 1

#         print(yyyy,mm,dd)
#         if yyyy < tyyyy:
#             answer.append(i+1)
#             continue
#         if mm < tmm:
#             answer.append(i+1)
#             continue
#         if dd < tdd:
#             answer.append(i+1)
#             continue

    return answer