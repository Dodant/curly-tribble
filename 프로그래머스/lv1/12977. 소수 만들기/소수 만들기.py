from itertools import combinations

def solution(numbers):
    answer = []
    for i in combinations(numbers, 3):
        if is_prime(sum(i)):
            answer.append(sum(i))
    return len(answer)
            
def is_prime(n):
    for i in range(2, n):
        if n % i == 0:
            return False
    return True
    
    