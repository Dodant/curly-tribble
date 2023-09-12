a, b = map(int, input().strip().split())

def uclid(a, b):
    if b == 0: return a
    return uclid(b, a%b)

lcm = uclid(a, b)
print((a // lcm) * (b // lcm) * lcm)