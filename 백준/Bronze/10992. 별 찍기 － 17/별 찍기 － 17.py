n = int(input())

if n == 1:
    print('*')
    exit()
print(' '*(n-1), '*', sep='')
for i in range(2, n):
    print(' '*(n-i), '*', ' '*(2*(i-1)-1), '*', sep='')
print('*'*(n*2-1))