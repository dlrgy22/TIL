def solution(n, times):
    times = sorted(times)
    min = 1
    max = times[-1] * n

    while True:
        answer = (min + max) // 2
        people = 0
        for time in times:
            people += answer // time

        if max == min:
            break
        move_num = (max - min) // 2

        if people < n:
            if move_num != 0:
                min += move_num
            else:
                min += 1
        else:
            if move_num != 0:
                max -= move_num
            else:
                max -= 1
    return answer


n = 6
times = [7, 10]
print(solution(n, times))