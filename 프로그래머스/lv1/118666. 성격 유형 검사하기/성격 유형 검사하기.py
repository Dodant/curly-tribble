def solution(survey, choices):
    answer = ''
    score = [3, 2, 1, 0, 1, 2, 3]
    score_dict = {'R':0, 'T':0,
                  'C':0, 'F':0,
                  'J':0, 'M':0,
                  'A':0, 'N':0}
    for s, c in zip(survey, choices):
        if c in [1, 2, 3]:
            score_dict[s[0]] += score[c-1]
        if c in [5, 6, 7]:
            score_dict[s[1]] += score[c-1]
    
    if score_dict['R'] >= score_dict['T']:
        answer += 'R'
    elif score_dict['R'] < score_dict['T']:
        answer += 'T'
        
    if score_dict['C'] >= score_dict['F']:
        answer += 'C'
    elif score_dict['C'] < score_dict['F']:
        answer += 'F'
        
    if score_dict['J'] >= score_dict['M']:
        answer += 'J'
    elif score_dict['J'] < score_dict['M']:
        answer += 'M'
        
    if score_dict['A'] >= score_dict['N']:
        answer += 'A'
    elif score_dict['A'] < score_dict['N']:
        answer += 'N'
        
    return answer