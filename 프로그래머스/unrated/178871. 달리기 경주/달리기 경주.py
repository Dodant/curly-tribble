def solution(players, callings):
    answer = {}
    for i, item in enumerate(players): answer[item] = i
    for player in callings:
        fr_idx, bk_idx = answer[player] - 1, answer[player]
        answer[player] = fr_idx
        answer[players[fr_idx]] = bk_idx
        players[fr_idx], players[bk_idx] = players[bk_idx], players[fr_idx]
    return players