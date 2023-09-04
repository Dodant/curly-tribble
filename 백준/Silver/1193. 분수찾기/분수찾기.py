X = int(input())

i = 1
c = 0
while True:
    if X > i * (i + 1) // 2:
        i += 1
    else:
        c = i * (i + 1) // 2 - X
        break 

if i % 2 == 0:
    print(f'{i - c}/{c + 1}')
else:
    print(f'{c + 1}/{i - c}')