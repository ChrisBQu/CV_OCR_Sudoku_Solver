import cv2

from image_processing import *
from sudoku import *

def runProcess(filename, debug=False):
    image = cv2.imread(filename)
    preprocessed_image = preprocessImage(image, debug)
    puzzle_cont = isolatePuzzle(preprocessed_image, debug)
    puzzle_mat = warpPuzzle(preprocessed_image, puzzle_cont)
    buckets = detectNumbers(puzzle_mat)
    solved = solveSudoku(buckets)
    if debug:
        cv2.imshow("Image", image)
        cv2.imshow("Puzzle Mat", puzzle_mat)
        print(solved)

    print("\nSolution: ")
    printSudoku(solved)

    while True:
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'): break


runProcess("tests/0.jpg", True)

cv2.destroyAllWindows()
