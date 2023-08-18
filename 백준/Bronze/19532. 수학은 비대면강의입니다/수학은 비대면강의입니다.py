a, b, c, d, e, f = map(int, input().split())
if a == 0:
    Y = c // b
    X = (f - e*Y) // d
    print(X, Y)
    exit()
Y = (a*f - c*d) // (a*e - d*b)
X = (c - b*Y) // a
print(X, Y)