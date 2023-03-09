def solution(A, B):
    repo = []
    for i in range(len(A)):
        repo.append(A[len(A)-i:] + A[:len(A)-i])
    if B in repo:
        return repo.index(B)
    return -1