import sys

def solution(paper, location, answer):
    prev_value = paper[location[0][0]][location[1][0]]
    for i in range(location[0][0], location[0][1]):
        for j in range(location[1][0], location[1][1]):
            if prev_value != paper[i][j]:
                length = (location[0][1] - location[0][0]) // 3
                for y in range(3):
                    for x in range(3):
                        next_location = [[location[0][0] + length*y, location[0][0] + length * (y + 1)], [location[1][0] + length*x, location[1][0] + length * (x + 1)]]
                        solution(paper, next_location, answer)

                return
    answer[prev_value + 1] += 1



n = int(sys.stdin.readline())
paper = [list(map(int, sys.stdin.readline().split())) for i in range(n)]
answer = [0, 0, 0]
location = [[0, n], [0, n]]
solution(paper, location, answer)
for element in answer:
    print(element)