import word_scorer # c++ library
import time
import pytesseract # opticaal character recognition
import cv2 # image processing
import numpy as np # image processing
from PIL import Image, ImageGrab

pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

# bonuses = [((2,0), "TL"),
#            ((3,2), "2x")]

def parse_image_to_grid(image):
    global pink_square_index
    global yellow_circle_index
    global orange_circle_index
    
    print("applying post processing to image")
    # Load the image, filter out pink, and convert it to grayscale
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    color_to_filter = np.array([235, 35, 255])
    lower_range = color_to_filter - np.array([10, 10, 10])
    upper_range = color_to_filter + np.array([10, 10, 10])
    mask = cv2.inRange(image, lower_range, upper_range)
    image_mask = image.copy()
    image_mask[mask != 0] = [0, 0, 0]
    gray = cv2.cvtColor(image_mask, cv2.COLOR_BGR2GRAY)
    # cv2.imshow("gray", gray)

    # Apply adaptive thresholding to binarize the image
    thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
    # cv2.imshow("thresh", thresh)

    # Apply opening to remove small noise
    open_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, open_kernel, iterations=3)
    # cv2.imshow("opening", opening)
    open_kernel_contour = cv2.getStructuringElement(cv2.MORPH_RECT, (4, 4))
    opening_contour = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, open_kernel_contour, iterations=3)
    dilate_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (4, 4))
    opening_contour = cv2.dilate(opening_contour, dilate_kernel, iterations=8)
    # cv2.imshow("opening_contour", opening_contour)

    # Find the contours of the black lines
    contours, _ = cv2.findContours(opening_contour, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    # Get inner most contours
    contours = sorted(contours, key=cv2.contourArea, reverse=True)[1:]
    # Sort contours from left to right top to bottom
    contours = sorted(contours, key=lambda contour: contour[0][0][1] * 5 + contour[0][0][0])
    # Move all contours up and left by percentage of width and height
    percent_to_move = 0.15
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        move_x = int(w * percent_to_move)
        move_y = int(h * percent_to_move)
        contour[:, :, 0] -= move_x
        contour[:, :, 1] -= move_y

    # Filter out everything except pink_to_filter
    image_mask = image.copy()
    image_mask[mask == 0] = [0, 0, 0]
    # Binarize the image
    image_mask = cv2.cvtColor(image_mask, cv2.COLOR_BGR2GRAY)
    image_mask = cv2.threshold(image_mask, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
    image_mask = cv2.bitwise_not(image_mask)
    # cv2.imshow("pink_mask", pink_image)
    # Apply opening to remove small noise
    open_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    pink_opening = cv2.morphologyEx(image_mask, cv2.MORPH_OPEN, open_kernel, iterations=2)
    # cv2.imshow("pink_opening", pink_opening)
    pink_contours, _ = cv2.findContours(pink_opening, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    if (len(pink_contours) > 0):
        # Calculate center of largest bounding box of pink contours
        pink_contours = sorted(pink_contours, key=cv2.contourArea, reverse=True)
        x, y, w, h = cv2.boundingRect(pink_contours[0])
        contour_center = (x + w/2, y + h/2)
        # Find which contour the center of pink_contour is in
        pink_square_index = -1
        for i in range(len(contours)):
            if cv2.pointPolygonTest(contours[i], contour_center, False) == 1:
                pink_square_index = i
                break
        print(f"2x square index: {pink_square_index}")

    # Filter out everything except yellow
    color_to_filter = np.array([163, 245, 252])
    lower_range = color_to_filter - np.array([10, 10, 10])
    upper_range = color_to_filter + np.array([10, 10, 10])
    mask = cv2.inRange(image, lower_range, upper_range)
    image_mask = image.copy()
    image_mask[mask == 0] = [0, 0, 0]
    # Binarize the image
    image_mask = cv2.cvtColor(image_mask, cv2.COLOR_BGR2GRAY)
    image_mask = cv2.threshold(image_mask, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
    image_mask = cv2.bitwise_not(image_mask)
    # Dilate the image to make the circles bigger
    open_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    image_mask = cv2.dilate(image_mask, open_kernel, iterations=2)
    # cv2.imshow("yellow_mask", yellow_image)
    # Find the contours of the yellow circles
    yellow_contours, _ = cv2.findContours(image_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    if (len(yellow_contours) > 0):
        yellow_contour = sorted(yellow_contours, key=cv2.contourArea, reverse=True)[0]
        x, y, w, h = cv2.boundingRect(yellow_contour)
        contour_center = (x + w/2, y + h/2)
        # move center by 3/4 of width and height of white square contour width and height
        x, y, w, h = cv2.boundingRect(contours[0])
        move_x = int(w * 1/2)
        move_y = int(h * 1/2)
        contour_center = (contour_center[0] + move_x, contour_center[1] + move_y)
        # Find which contour the center of yellow_contour is in
        for i in range(len(contours)):
            if cv2.pointPolygonTest(contours[i], contour_center, False) == 1:
                yellow_circle_index = i
                break
        print(f"DL square index: {yellow_circle_index}")
        # draw contours on image each with a different color
        cv2.drawContours(image, yellow_contours, -1, (255, 0, 0), 2)
    else:
        color_to_filter = np.array([64, 112, 214])
        lower_range = color_to_filter - np.array([10, 10, 10])
        upper_range = color_to_filter + np.array([10, 10, 10])
        mask = cv2.inRange(image, lower_range, upper_range)
        image_mask = image.copy()
        image_mask[mask == 0] = [0, 0, 0]
        # Binarize the image
        image_mask = cv2.cvtColor(image_mask, cv2.COLOR_BGR2GRAY)
        image_mask = cv2.threshold(image_mask, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
        image_mask = cv2.bitwise_not(image_mask)
        # Dilate the image to make the circles bigger
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        image_mask = cv2.dilate(image_mask, kernel, iterations=2)
        # cv2.imshow("orange_mask", orange_image)
        # Find the contours of the orange circles
        orange_contours, _ = cv2.findContours(image_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
        if (len(orange_contours) > 0):
            orange_contour = sorted(orange_contours, key=cv2.contourArea, reverse=True)[0]
            x, y, w, h = cv2.boundingRect(orange_contour)
            contour_center = (x + w/2, y + h/2)
            # move center by 3/4 of width and height of white square contour width and height
            x, y, w, h = cv2.boundingRect(contours[0])
            move_x = int(w * 1/2)
            move_y = int(h * 1/2)
            contour_center = (contour_center[0] + move_x, contour_center[1] + move_y)
            # Find which contour the center of orange_contour is in
            for i in range(len(contours)):
                if cv2.pointPolygonTest(contours[i], contour_center, False) == 1:
                    orange_circle_index = i
                    break
            print(f"TL square index: {orange_circle_index}")
            # draw contours on image each with a different color
            cv2.drawContours(image, orange_contours, -1, (255, 0, 0), 2)
        else:
            print("DL or TL not found")

    # draw contours on image each with a different color
    # for i in range(len(contours)):
    #     cv2.drawContours(image, [contours[i]], -1, (0, 0, 25), 2)
    #     cv2.putText(image, str(i), (contours[i][0][0][0], contours[i][0][0][1]), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
    # cv2.imshow("contours", image)
    # cv2.waitKey(0)

    # pytesseraact on each contour to get the letter in the box and add it to grid
    # make grid of empty chars
    print("parsing letters")
    grid = [['' for j in range(5)] for i in range(5)]
    for i in range(5):
        for j in range(5):
            x, y, w, h = cv2.boundingRect(contours[i*5 + j])
            scale_image = opening[y:y+h, x:x+w]
            scale_percent = 70
            width = int(scale_image.shape[1] * scale_percent / 100)
            height = int(scale_image.shape[0] * scale_percent / 100)
            dim = (width, height)
            resized = cv2.resize(scale_image, dim, interpolation = cv2.INTER_AREA)
            letter = pytesseract.image_to_string(resized, config="--psm 10 --oem 3 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ").strip().lower()
            # cv2.imshow("resized", resized)
            # cv2.waitKey(0)
            # if letter is not exactly one letter, try again but 10% smaller
            if (len(letter) != 1):
                # print(f"failed to parse letter {i*5 + j}, trying again")
                scale_percent = 300
                width = int(scale_image.shape[1] * scale_percent / 100)
                height = int(scale_image.shape[0] * scale_percent / 100)
                dim = (width, height)
                resized = cv2.resize(scale_image, dim, interpolation = cv2.INTER_LINEAR)
                resized = cv2.blur(resized, (5, 5))
                letter = pytesseract.image_to_string(resized, config="--psm 10 --oem 3 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ").strip().lower()
                if (len(letter) != 1):
                    # cv2.imshow("resized", resized)
                    # cv2.waitKey(0)
                    print(f"failed to parse letter {i*5 + j} again, asking user for input")
                    # Request user input for letter and show image so they know what letter it is
                    error_image = Image.fromarray(resized)
                    error_image.show()
                    letter = input(f"What letter is in box {i*5 + j}? ").strip().lower()
            grid[i][j] = letter
    # cv2.waitKey(0)
    return grid

start = time.time()
# input_image = cv2.imread("example_imageDL.png")
# input_image = cv2.cvtColor(input, cv2.COLOR_BGR2RGB)
# get input from clipboard
input_image = np.array(ImageGrab.grabclipboard())
pink_square_index = -1
yellow_circle_index = -1
orange_circle_index = -1
grid = parse_image_to_grid(input_image)
bonuses = []
for i in range(5):
    for j in range(5):
        if (i*5 + j) == pink_square_index:
            bonuses.append(((i, j), "2x"))
        if (i*5 + j) == yellow_circle_index:
            bonuses.append(((i, j), "DL"))
        elif (i*5 + j) == orange_circle_index:
            bonuses.append(((i, j), "TL"))
print(f"{bonuses=}")
print(grid)
print(f"parsing time: {time.time() - start} seconds")
start_calc = time.time()
print("calculating scores...")
scores = word_scorer.getWordScores(grid, bonuses)
for i in range(10):
    print(f"{i+1}. {scores[i][0]}:\t{scores[i][1]}\t{scores[i][2]}")
print(f"calculating time: {time.time() - start_calc} seconds")
print(f"total time: {time.time() - start} seconds")