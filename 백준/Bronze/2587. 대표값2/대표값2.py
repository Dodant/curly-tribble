ls = [0] * 5
for i in range(5):
    ls[i] = int(input())
ls.sort()   
print(sum(ls)//5)
print(ls[2])