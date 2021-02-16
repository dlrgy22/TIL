import sys

def find_blank(sudoku):
    blank_list = []
    for i in range(9):
        for j in range(9):
            if sudoku[i][j] == 0:
                blank_list.append([i, j])
    return blank_list

def check_row(col, sudoku, check_num):
    for i in range(9):
        if sudoku[col][i] == check_num:
            return False
    return True

def check_col(row, sudoku, check_num):
    for i in range(9):
        if sudoku[i][row] == check_num:
            return False
    return True

def check_box(col, row, sudoku, check_num):
    for i in range(3):
        for j in range(3):
            if sudoku[col + i][row + j] == check_num:
                return False
    return True

def print_sudoku(sudoku):
    for i in range(9):
        for j in range(9):
            print(sudoku[i][j], end=' ') if j != 8 else print(sudoku[i][j])

def dfs(sudoku, blank_list, idx):
    location = blank_list[idx]

    for check_num in range(1, 10):
        if check_row(location[0], sudoku, check_num) and check_col(location[1], sudoku, check_num) and check_box(location[0]//3 * 3, location[1]//3 * 3, sudoku, check_num):
            sudoku[location[0]][location[1]] = check_num
            if idx == len(blank_list) - 1:
                print_sudoku(sudoku)
                sys.exit()
            dfs(sudoku, blank_list, idx + 1)
            sudoku[location[0]][location[1]] = 0
            
if __name__ == "__main__":
    sudoku = [list(map(int, sys.stdin.readline().split())) for i in range(9)]
    blank_list = find_blank(sudoku)
    dfs(sudoku, blank_list, 0)