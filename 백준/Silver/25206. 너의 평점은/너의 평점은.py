DD = {'A+':4.5, 'A0':4.0, 'B+':3.5, 'B0':3.0, 'C+':2.5, 'C0':2.0, 'D+':1.5, 'D0':1.0, 'F':0.0}

sum_hj = 0
sum_cr = 0
for _ in range(20):
    ls = [input().split()]
    hj, cr = float(ls[0][1]), ls[0][2]
    
    if cr == "P": continue 
    sum_hj += hj
    sum_cr += DD[cr] * hj
    
print(sum_cr / sum_hj)