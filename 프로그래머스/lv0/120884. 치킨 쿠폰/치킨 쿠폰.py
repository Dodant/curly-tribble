def solution(chicken):
    eat = []
    rest = []
    while True:
        moc, nam = chicken // 10, chicken % 10
        eat.append(moc)
        rest.append(nam)
        chicken = moc
        if chicken == 0:
            break
    eat.append(sum(rest)//10)
    eat.append((sum(rest)%10 + sum(rest)//10)//10)
    return sum(eat)