#-- Takes a 1-Dimensional array of numbers representing a Sudoku puzzle,
#-- going rows first then columns, with 0s representing blank spots, and
#-- solves it. If it cannot be solved, returns the original array
def solveSudoku(puzzle_arr):
    def solve():
        for i in range(9):
            for j in range(9):
                if puzzle_arr[i * 9 + j] == 0:
                    for num in range(1, 10):
                        if isValidSudoku(puzzle_arr, i, j, num):
                            puzzle_arr[i * 9 + j] = num
                            if solve(): return True
                            puzzle_arr[i * 9 + j] = 0
                    return False # No valid solutions
        return True 
    done = solve()
    if done: return puzzle_arr

#-- Helper function returns True if puzzle is valid, or False if it is invalid
def isValidSudoku(board, row, col, num):
    # Check the row
    for i in range(9):
        if board[row * 9 + i] == num:
            return False

    # Check the column
    for i in range(9):
        if board[i * 9 + col] == num:
            return False

    # Check the 3x3 subgrid
    start_row = (row // 3) * 3
    start_col = (col // 3) * 3
    for i in range(3):
        for j in range(3):
            if board[(start_row + i) * 9 + (start_col + j)] == num:
                return False

    return True

#-- Print board to console
def printSudoku(board):
    for i in range(9):
        # Print horizontal dashed line every 3 rows
        if i % 3 == 0 and i != 0:
            print("-" * 21)
        
        row = ""
        for j in range(9):
            # Print vertical dashed line every 3 columns
            if j % 3 == 0 and j != 0:
                row += "| "
            
            # Append the number (replace 0 with a dot for better clarity)
            row += str(board[i * 9 + j]) if board[i * 9 + j] != 0 else "."
            row += " "
        
        print(row)
            
