import copy

def solution(board):
    n = len(board)
    big_board = [[0]*(n+2)]
    for i in board: big_board.append([0] + i + [0])
    big_board.append([0]*(n+2))
    
    alt_board = copy.deepcopy(big_board)
    for i in range(1,n+1):
        for j in range(1,n+1):
            if big_board[i][j] == 1:
                alt_board[i-1][j-1] = 2
                alt_board[i-1][j]   = 2
                alt_board[i-1][j+1] = 2
                alt_board[i][j-1]   = 2
                alt_board[i][j]     = 2
                alt_board[i][j+1]   = 2
                alt_board[i+1][j-1] = 2
                alt_board[i+1][j]   = 2
                alt_board[i+1][j+1] = 2
    
    cnt = 0
    for i in range(1,n+1):
        for j in range(1,n+1):
            if alt_board[i][j] == 0:
                cnt += 1
    return cnt