def solution(keyinput, board):
    x_pos, y_pos = 0, 0
    x_lim, y_lim = (board[0]-1) // 2, (board[1]-1) // 2
    for i in keyinput:
        if i == 'left': x_pos -= 1
        if i == 'right': x_pos += 1
        if i == 'up': y_pos += 1
        if i == 'down': y_pos -= 1
        if x_pos > x_lim: x_pos -= 1
        if x_pos < -1 * x_lim: x_pos += 1
        if y_pos > y_lim: y_pos -= 1
        if y_pos < -1 * y_lim: y_pos += 1
    return [x_pos, y_pos]