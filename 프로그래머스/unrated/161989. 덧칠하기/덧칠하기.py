def solution(n, m, section):
    cursor = 0
    answer = 0
    coverage = section[cursor] - 1
    while cursor < len(section):
        if coverage > n-1:
            break
        if coverage < section[cursor]:
            coverage = section[cursor] + m - 1
            cursor += 1
            answer += 1
        else:
            cursor += 1
        
    return answer