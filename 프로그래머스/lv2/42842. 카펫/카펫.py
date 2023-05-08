def solution(brown, yellow):
    answer = []
    for i in range(brown + yellow, 2, -1):
        if (brown + yellow) % i == 0:
            if (brown - 2 * i) // 2 + 2 == (brown + yellow) // i:
                if ((brown + yellow) // i) <= 2:
                    continue
                answer.append(i)
                answer.append((brown + yellow) // i)
                break
    return answer