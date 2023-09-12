c, a = map(int, input().strip().split())
d, b = map(int, input().strip().split())

def uclid(a, b):
    if b == 0: return a
    return uclid(b, a%b)

lcm = uclid(a, b)
bj = c*(b//lcm) + d*(a//lcm)
bm = (a // lcm) * (b // lcm) * lcm

lcm = uclid(bj, bm)
bj //= lcm
bm //= lcm
print(bj, bm)