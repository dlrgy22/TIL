import sys

def solution(v):
    dic_x = {}
    dic_y = {}
    for v_element in v:
        if v_element[0] in dic_x:
            dic_x[v_element[0]] += 1
        else:
            dic_x[v_element[0]] = 1

        if v_element[1] in dic_y:
            dic_y[v_element[1]] += 1
        else:
            dic_y[v_element[1]] = 1

    for k, v in dic_x.items():
        if v == 1:
            x = k
    for k, v in dic_y.items():
        if v == 1:
            y = k

    answer = [x, y]
    return answer

v = [[1, 4], [3, 4], [3, 10]]
solution(v)