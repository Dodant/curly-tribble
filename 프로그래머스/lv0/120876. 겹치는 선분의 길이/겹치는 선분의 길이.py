def solution(lines):
    a = set(range(lines[0][0], lines[0][1]))
    b = set(range(lines[1][0], lines[1][1]))
    c = set(range(lines[2][0], lines[2][1]))
    return len(a&b | b&c | a&c)