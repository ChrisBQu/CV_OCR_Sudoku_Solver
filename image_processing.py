import cv2
import numpy as np
import easyocr

#-- Preprocess the image so that it is a clean black and white image
#-- that the puzzle can be extracted from
def preprocessImage(img, visualize=False):
    #-- Cnvert the image to grayscale, blur it, and apply adaptive thresholding
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.medianBlur(gray, 3)
    thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 15, 15)
    #-- Invert the image and apply morphological transformations to fill in small gaps in the lines
    inverted = cv2.bitwise_not(thresh)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    closed = cv2.dilate(inverted, kernel, iterations=1)
    if visualize: cv2.imshow("Preprocessed", closed)
    return closed

#-- Find the warped rectangles in the image, and then return the largest
#-- of these rectangles, assuming it meets a minimum size requirement
def isolatePuzzle(image, visualize=False):
    CONTOUR_MINIMUM_SIZE = 10000
    color = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    blurred = cv2.GaussianBlur(image, (5,5), 0) #-- Blurring the puzzle can help with warped "wavy" edges
    #-- Apply canny edge detection, and then find the contours in the image
    edges = cv2.Canny(blurred, 50, 150)
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)
    for contour in contours:
        #-- Simplify the contour while retaining its shape
        epsilon = 0.02 * cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, epsilon, True)
        area = cv2.contourArea(contour)
        #-- Disregard small contours, or contours without 4 vertices
        if len(approx) == 4 and area >= CONTOUR_MINIMUM_SIZE:
            if visualize:
                cv2.drawContours(color, [approx], -1, (0, 0, 255), 2)
                cv2.imshow("Isolated Puzzle", color)
            return approx
    return None

#-- Take an image, and the contour defining where the puzzle is located on the page, and warp
#-- that portion of the image such that the puzzle is in a straight-on view, in its own mat
def warpPuzzle(image, contour):
    # Convert the approximated contour to a 4-point format, in the right order
    rect = np.array(contour).reshape(4, 2)
    rect = sortRectanglePoints(rect)
    
    # Compute the width and height of the new image
    (tl, tr, br, bl) = rect
    widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    maxWidth = max(int(widthA), int(widthB))
    heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
    maxHeight = max(int(heightA), int(heightB))
    
    # Destination points for the warp
    dst = np.array([
        [0, 0],
        [maxWidth - 1, 0],
        [maxWidth - 1, maxHeight - 1],
        [0, maxHeight - 1]], dtype="float32")
    
    #-- Apply transformation and return
    transform_matrix = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(image, transform_matrix, (maxWidth, maxHeight))
    return warped

#-- Helper function for warpPuzzle function
#-- This makes sure the four coordinates of the puzzle's bounding rectangle
#-- are in the right order.
def sortRectanglePoints(pts):
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    return rect

#-- Function takes an image (in the format produced by warpPuzzle()), and returns an array of
#-- length 81 representing the Sudoku board that was read in from the image
def detectNumbers(img, verbose=False):

    # Enlarge and blur the image slightly for better detection
    copied = img.copy()
    height, width = img.shape[:2]
    copied = cv2.resize(img, (int(1.5*width) ,int(1.5*height)), interpolation=cv2.INTER_LINEAR)
    copied = cv2.blur(copied, (11,11))
    # Divide the image into 81 cells
    buckets = [0] * 81
    height, width = copied.shape[:2]
    cell_width = width // 9
    cell_height = height // 9

    # Make a temporary image file and read it with EasyOCR
    cv2.imwrite('temp_image.png', copied)
    reader = easyocr.Reader(['en'])
    result = reader.readtext('temp_image.png', allowlist='0123456789')

    for each in result:
        # Get the center coordinates of the detected text box, and use it to find the row/column of the digit
        center_x = int((each[0][0][0] + each[0][3][0]) / 2)
        center_y = int((each[0][0][1] + each[0][3][1]) / 2)
        col = center_x // cell_width
        row = center_y // cell_height
        # Store the detected value in the appropriate bucket
        bucket_index = row * 9 + col
        value = int(each[1])
        buckets[bucket_index] = value
        if verbose: print(f"Detected value '{value}' at bucket {bucket_index} (row {row}, col {col})")
    return buckets 
