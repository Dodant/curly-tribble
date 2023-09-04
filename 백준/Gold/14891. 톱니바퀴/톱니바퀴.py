def rotate(gear, direction):
    return [gear[-1]] + gear[:-1] if direction == 1 else gear[1:] + [gear[0]]

gear = [list(map(int, list(*input().split()))) for _ in range(4)]
N = int(input().strip())
state = [list(map(int, input().split())) for _ in range(N)]

for i, direction in state:
    r1, l2, r2, l3, r3, l4 = gear[0][2], gear[1][6], gear[1][2], gear[2][6], gear[2][2], gear[3][6]
    
    if i == 1:
        if direction == 1:
            gear[0] = rotate(gear[0], 1)
            if r1 != l2:
                gear[1] = rotate(gear[1], -1)
                if r2 != l3:
                    gear[2] = rotate(gear[2], 1)
                    if r3 != l4:
                        gear[3] = rotate(gear[3], -1)
        elif direction == -1:
            gear[0] = rotate(gear[0], -1)
            if r1 != l2:
                gear[1] = rotate(gear[1], 1)
                if r2 != l3:
                    gear[2] = rotate(gear[2], -1)
                    if r3 != l4:
                        gear[3] = rotate(gear[3], 1)
    elif i == 2:
        if direction == 1:
            gear[1] = rotate(gear[1], 1)
            if r2 != l3:
                gear[2] = rotate(gear[2], -1)
                if r3 != l4:
                    gear[3] = rotate(gear[3], 1)
            if r1 != l2:
                gear[0] = rotate(gear[0], -1)
        elif direction == -1:
            gear[1] = rotate(gear[1], -1)
            if r2 != l3:
                gear[2] = rotate(gear[2], 1)
                if r3 != l4:
                    gear[3] = rotate(gear[3], -1)
            if r1 != l2:
                gear[0] = rotate(gear[0], 1)
    elif i == 3:
        if direction == 1:
            gear[2] = rotate(gear[2], 1)
            if r3 != l4:
                gear[3] = rotate(gear[3], -1)
            if r2 != l3:
                gear[1] = rotate(gear[1], -1)
                if r1 != l2:
                    gear[0] = rotate(gear[0], 1)
        elif direction == -1:
            gear[2] = rotate(gear[2], -1)
            if r3 != l4:
                gear[3] = rotate(gear[3], 1)
            if r2 != l3:
                gear[1] = rotate(gear[1], 1)
                if r1 != l2:
                    gear[0] = rotate(gear[0], -1)
    elif i == 4:
        if direction == 1:
            gear[3] = rotate(gear[3], 1)
            if r3 != l4:
                gear[2] = rotate(gear[2], -1)
                if r2 != l3:
                    gear[1] = rotate(gear[1], 1)
                    if r1 != l2:
                        gear[0] = rotate(gear[0], -1)
        elif direction == -1:
            gear[3] = rotate(gear[3], -1)
            if r3 != l4:
                gear[2] = rotate(gear[2], 1)
                if r2 != l3:
                    gear[1] = rotate(gear[1], -1)
                    if r1 != l2:
                        gear[0] = rotate(gear[0], 1)
                        
print(int('0b'+str(gear[3][0])+str(gear[2][0])+str(gear[1][0])+str(gear[0][0]), 2))