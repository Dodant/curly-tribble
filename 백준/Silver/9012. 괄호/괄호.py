T = int(input())

for _ in range(T):
    S = input()
    ST = []
    for s in S:
        if s == '(':
            ST.append(s)
        elif s == ')':
            if len(ST) == 0:
                ST.append(s)
                break
            else:
                ST.pop()
    print('YES' if len(ST) == 0 else 'NO')