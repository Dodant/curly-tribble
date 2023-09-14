def solution(wallpaper):
    x_ls = []
    y_ls = []
    for i in range(len(wallpaper)):
        for j in range(len(wallpaper[0])):
            if wallpaper[i][j] == '#':
                x_ls.append(i)
                y_ls.append(j)
    return [min(x_ls), min(y_ls), max(x_ls)+1, max(y_ls)+1]