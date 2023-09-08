ls = [*map(int, input().split())]
ls.sort()
x, y, z = ls

if x + y > z:
    print(x + y + z)
else:
    if x + y == z:
        print(x + y + z - 1)
    elif x + y < z:
        print(x + y + (x + y - 1))