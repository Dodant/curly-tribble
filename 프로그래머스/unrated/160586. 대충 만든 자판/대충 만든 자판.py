def solution(keymap, targets):
    answer = []
    for target in targets:
        click = 0
        for i in target:
            indexs = []
            for j in keymap:
                if i in j:
                    if j.find(i) == -1:
                        continue
                    indexs.append(j.find(i)+1)
            if indexs == []:
                click = -1
                break
            click += min(indexs)
        answer.append(click)
    return answer