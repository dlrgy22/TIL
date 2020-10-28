def make_dp(dp, n1, n2):
    for element1 in dp[n1]:
        for element2 in dp[n2]:
            add_num = element1 + element2
            dp[n1 + n2].add(add_num)

            add_num = abs(element1 - element2)
            dp[n1 + n2].add(add_num)

            if min(element1, element2) != 0:
                add_num = max(element1, element2) // min(element1, element2)
                dp[n1 + n2].add(add_num)

            add_num = element1 * element2
            dp[n1 + n2].add(add_num)



def solution(N, number):
    dp = [set() for i in range(9)]
    for i in range(1, 9):
        num = str(N) * i
        dp[i].add(int(num))

    for i in range(1, 9):
        for j in range(1, i + 1):
            if i + j <= 8:
                make_dp(dp, i, j)

    for i in range(1, 9):
        if number in dp[i]:
            return i

    return -1

N = 2
number = 11
print(solution(N, number))