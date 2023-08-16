S = input()

if S in ['P', 'PPAP']:
    print('PPAP')
else:
    F = []
    for i in S:
        F.append(i)
        if len(F) >= 4 and F[-4:] == ['P', 'P', 'A', 'P']:
            F[-4:] = ['P']
    if F == ['P' ]:
        print('PPAP')
    else:
        print('NP')