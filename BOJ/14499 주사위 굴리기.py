import sys
def move_dice(loc_x, loc_y, direction, state):
    if direction == 1:
        if loc_x + 1 >= n:
            return loc_x, loc_y, state
        loc_x += 1
        state = [state[1], dice[state[0]], state[0], state[3], state[4]]
    elif direction == 2:
        if loc_x - 1 < 0:
            return loc_x, loc_y, state
        loc_x -= 1
        state = [state[2], state[0], dice[state[0]], state[3], state[4]]

    elif direction == 3:
        if loc_y - 1 < 0:
            return loc_x, loc_y, state
        loc_y -= 1
        state = [state[3], state[1], state[2], dice[state[0]], state[0]]
    else:
        if loc_y + 1 >=  n:
            return loc_x, loc_y, state
        loc_y += 1
        state = [state[4], state[1], state[2], state[0], dice[state[0]]]

    if my_map[loc_y][loc_x] != 0:
        dice_element[dice[state[0]]] = my_map[loc_y][loc_x]
        my_map[loc_y][loc_x] = 0
    else:
        my_map[loc_y][loc_x] = dice_element[dice[state[0]]]
    print(dice_element[state[0]])
    return loc_x, loc_y, state

n, m, x, y, k = map(int, sys.stdin.readline().split())
my_map = [[0 for i in range(m)] for j in range(n)]
for i in range(n):
    map_element = list(map(int, sys.stdin.readline().split()))
    for j in range(len(map_element)):
        my_map[i][j] = map_element[j]
order = list(map(int, sys.stdin.readline().split()))
dice = [0, 6, 5, 4, 3, 2, 1]
dice_element = [0 for i in range(7)]
state = [1, 4, 3, 5, 2]
for order_element in order:
    #print(x, y)
    x, y, state = move_dice(x, y, order_element, state)