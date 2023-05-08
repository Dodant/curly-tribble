def solution(n, words):
    answer = []
    
    for idx, item in enumerate(words):
        if item in answer:
            return [idx%n+1, idx//n+1]
        else: 
            answer.append(item)
    
    for idx, item in enumerate(words):
        if idx == 0:
            continue
        if item[0] != words[idx-1][-1]:
            return [idx%n+1, idx//n+1]
            
    return [0, 0]