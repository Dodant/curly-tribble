S = input()
T = input()

def first_condition_check(T):
    if T[-1] == 'A':
        Tr = T[:-1]
        return True, Tr
    return False, T
    
def second_condition_check(T):
    if T[-1] == 'B':
        Tr = T[:-1][::-1]    
        return True, Tr
    return False, T

while True:
    r1, t1 = first_condition_check(T)
    r2, t2 = second_condition_check(T)
    
    if t1 == '' or t2 == '':
        print(0)
        break
    
    if r1:
        T = t1
    elif r2:
        T = t2

    if S == T:
        print(1)
        break    
            
    if not r1 and not r2:
        print(0)
        break
    