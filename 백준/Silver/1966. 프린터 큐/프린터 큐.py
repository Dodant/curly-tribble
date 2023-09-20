T = int(input())

for i in range(T):
    N, M = map(int, input().split())
    st = []
    answer = []
    q = list(map(int, input().split()))
    tq = [(i, 0) for i in q]
    tq[M] = (tq[M][0], 1)
    cnt = 0
    while tq:
        if max(tq)[0] == tq[0][0]:
            cpn = tq.pop(0)
            if cpn[1] == 1:
                print(cnt+1)
                break
            answer.append(cpn)
            cnt += 1
        else:
            tq.append(tq.pop(0))