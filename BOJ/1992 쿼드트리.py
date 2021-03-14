import sys

def solution(image,x_range, y_range, answer):
    length = x_range[1] - x_range[0]
    pixel = image[y_range[0]][x_range[0]]
    for y in range(y_range[0], y_range[1]):
        for x in range(x_range[0], x_range[1]):
            if pixel != image[y][x]:
                answer.append("(")
                for i in range(2):
                    for j in range(2):
                        next_x_range = [x_range[0] + length // 2 * j, x_range[0] + length // 2 * (j + 1)]
                        next_y_range = [y_range[0] + length // 2 * i, y_range[0] + length // 2 * (i + 1)]
                        solution(image, next_x_range, next_y_range, answer)
                answer.append(")")
                return
    answer.append(pixel)
    
n = int(sys.stdin.readline())
image = [list(sys.stdin.readline()[:-1]) for i in range(n)]
answer = []
x_range = [0, n]
y_range = [0, n]
solution(image, x_range, y_range, answer)
for element in answer:
    print(element,end="")