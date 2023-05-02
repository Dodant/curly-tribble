def solution(todo_list, finished):
    return [item for idx, item in enumerate(todo_list) if not finished[idx]]