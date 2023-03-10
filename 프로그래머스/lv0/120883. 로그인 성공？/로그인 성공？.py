def solution(id_pw, db):
    if id_pw in db:
        return 'login'
    for i in db:
        if id_pw[0] in i:
            return 'wrong pw'
    return 'fail'