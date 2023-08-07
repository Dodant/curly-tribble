def solution(lines):
    points = set()
    small_x, small_y = float('inf'), float('inf')
    big_x, big_y = -float('inf'), -float('inf')
    n = len(lines)
    
    for i in range(n):
        a, b, e = lines[i]
        for j in range(i + 1, n):
            c, d, f = lines[j]
            if a * d == c * b: continue
            x = (b * f - d * e) / (a * d - c * b)
            y = (a * f - c * e) / (a * d - c * b)
            if x != int(x) or y != int(y): continue
            x, y = int(x), int(y)
            if small_x > x: small_x = x
            if small_y > y: small_y = y
            if big_x < x: big_x = x
            if big_y < y: big_y = y
            
            points.add((int(x), int(y)))
                
    answer = ["." * (big_x - small_x + 1) for i in range(big_y - small_y + 1)]

    for x, y in points:
        x = x - small_x
        y = y - small_y
        answer[y] = ''.join([answer[y][:x],"*",answer[y][x+1:]])
        
    return answer 
