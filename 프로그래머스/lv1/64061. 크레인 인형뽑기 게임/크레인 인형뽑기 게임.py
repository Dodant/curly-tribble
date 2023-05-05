def solution(board, moves):
    answer = 0
    bag = []
    for i in moves:
        i -= 1
        for j in range(len(board)):
            if board[j][i] == 0:
                continue
            else:
                bag.append(board[j][i])
                board[j][i] = 0
                break
        if len(bag) >= 2:
            if bag[-1] == bag[-2]:
                bag = bag[:-2]
                answer += 2
    return answer