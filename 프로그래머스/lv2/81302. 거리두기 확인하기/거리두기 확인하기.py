def solution(places):
    answer = []
    
    for place in places:
        base = []
        for i in place: base.append(list(i))
        
        for i in range(25):
            result = True
            if base[i // 5][i % 5] == 'P':
                base, result = cross(base, i // 5, i % 5)
            if result == False:
                answer.append(0)
                break
        if result:
            answer.append(1)
    return answer



def cross(place, i, j):
    dx = [0, 0, -1, 1]
    dy = [-1, 1, 0, 0]
    
    for k in range(4):
        if i+dx[k] < 0 or i+dx[k] > 4 or j+dy[k] < 0 or j+dy[k] > 4:
            continue
        if place[i+dx[k]][j+dy[k]] == 'O':
            place[i+dx[k]][j+dy[k]] = 'Q'
        elif place[i+dx[k]][j+dy[k]] == 'X':
            continue
        elif place[i+dx[k]][j+dy[k]] == 'Q':
            return place, False
        elif place[i+dx[k]][j+dy[k]] == 'P':
            return place, False
    return place, True