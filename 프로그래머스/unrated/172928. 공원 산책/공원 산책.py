def solution(park, routes):
    start = [0, 0]
    park = [list(i) for i in park]
    
    for i in range(len(park)):
        for j in range(len(park[0])):
            if park[i][j] == 'S':
                start = [i, j]
                park[i][j] = 0
            if park[i][j] == 'O':
                park[i][j] = 0
            if park[i][j] == 'X':
                park[i][j] = 1
                
    dx = [0, 0, 1, -1]
    dy = [1, -1, 0, 0]
    direc = ['E', 'W', 'S', 'N']
    
    for i in routes:
        direction, distance = i.split()
        distance = int(distance)
        direction_x, direction_y = dx[direc.index(direction)], dy[direc.index(direction)]
        
        if 0 <= start[0] + direction_x * distance < len(park) and \
            0 <= start[1] + direction_y * distance < len(park[0]):
                for k in range(1, distance + 1):
                    if k == distance and park[start[0] + direction_x * k][start[1] + direction_y * k] == 0:
                        start[0] += direction_x * distance
                        start[1] += direction_y * distance
                    elif park[start[0] + direction_x * k][start[1] + direction_y * k] == 0:
                        continue
                    elif park[start[0] + direction_x * k][start[1] + direction_y * k] == 1:
                        break

    return start
