def tilt(a, b):
	return abs((a[1] - b[1]) / (a[0] - b[0]))

def solution(dots):
    if tilt(dots[0], dots[1]) == tilt(dots[2], dots[3]) or \
    	tilt(dots[0], dots[2]) == tilt(dots[1], dots[3]) or \
        tilt(dots[0], dots[3]) == tilt(dots[1], dots[2]): return 1
    return 0