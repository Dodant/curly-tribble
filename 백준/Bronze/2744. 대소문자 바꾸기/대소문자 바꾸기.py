n = input()
b = []
for i in n:
    if i.isupper():
        b.append(i.lower())
    else:
        b.append(i.upper())
print(''.join(b))