T = input()
ls = ['', '', '', 'ABC', 'DEF', 'GHI', 'JKL', 'MNO', 'PQRS', 'TUV', 'WXYZ']
print(sum([ls.index(j) for i in T for j in ls if i in j]))