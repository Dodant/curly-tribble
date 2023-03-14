def solution(emergency):
    answer = [0]*len(emergency)
    for i, t in enumerate(reversed(sorted(emergency))):
        answer[emergency.index(t)] = i+1
    return answer