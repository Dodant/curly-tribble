def solution(numbers, hand):
    answer = ''
    pad = {1: (0, 0), 2: (0, 1), 3: (0, 2),
           4: (1, 0), 5: (1, 1), 6: (1, 2),
           7: (2, 0), 8: (2, 1), 9: (2, 2),
           '*': (3, 0), 0: (3, 1), '#': (3, 2)}
    lf, rf = pad['*'], pad['#']
    
    for i in numbers:
        if i in [1, 4, 7]:
            answer += 'L'
            lf = pad[i]
        elif i in [3, 6, 9]:
            answer += 'R'
            rf = pad[i]
        else:
            ld = abs(lf[0] - pad[i][0]) + abs(lf[1] - pad[i][1])
            rd = abs(rf[0] - pad[i][0]) + abs(rf[1] - pad[i][1])
            if ld > rd:
                answer += 'R'
                rf = pad[i]
            elif ld < rd:
                answer += 'L'
                lf = pad[i]
            elif ld == rd: 
                if hand == 'right':
                    answer += 'R'
                    rf = pad[i]
                else:
                    answer += 'L'
                    lf = pad[i]
    
    return answer