def solution(angle):
    if 0 < angle < 90: return 1
    if 90 == angle: return 2
    if 90 < angle < 180: return 3
    if 180 == angle: return 4