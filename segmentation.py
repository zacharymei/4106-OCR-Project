import numpy as np
import cv2
import matplotlib.pyplot as plt
import os
import shutil
import sys


# Threshold for Row Segmentation
# Make it large if row's height too short
# Range 0.9990 to 0.9999
tsRow = 0.9999

# Threshold for Character Segmentation
# Make it large if character not fully displayed
# Range 0.93 to 0.98
tsChar = 0.95


# Row Segmentation (input Image, output Images[])

def segRow(input, threshold):

    height, width = input.shape
    white, black = 0, 0
    start, end = 0, 0
    point = []

    for i in range(height):
        sum = 0
        lastwhite = white
        lastblack = black
        for j in range(width):
            sum += input[i, j]

        black = black + 1 if sum < width*255*threshold else 0
        white = white + 1 if sum >= width*255*threshold else 0

        if(white < lastwhite):
            start = i
        if(black < lastblack):
            end = i

        if(end != 0):
            point.append((start, end))
            start, end = (0, 0)

    images = []
    for p in point:
        start = p[0]
        end = p[1]
        image = input[start:end]
        images.append(image)

    return images

# Character Segmentation (input Image, output, Images[])

def segChar(input, threshold):

    chars = segRow(np.rot90(input), threshold)
    output = [np.rot90(c, 3) for c in chars]
    output.reverse()
    maximized = [maximazation(c, threshold) for c in output]

    return maximized

# Maximum the character to border (input Image, output Image)

def maximazation(input, threshold):

    for rota in range(4):
        input = np.rot90(input)
        height, width = input.shape

        white = 0
        edge = 0

        point = []
        for i in range(height):
            sum = 0
            lastwhite = white
            for j in range(width):
                sum += input[i, j]

            if(sum < width*255*threshold):
                edge = i
                break

        output = input[edge: height]
        input = output

    return input


def seg(input, thresholdRow = tsRow, thresholdChar = tsChar):

    rows = segRow(input, thresholdRow)

    fullCharacters = []
    for r in rows:
        fullCharacters.append(segChar(r, thresholdChar))


    # Plot generation

    # Comment from here if you don't need the plot

    fig=plt.figure(figsize=(rows[0].shape[1]*0.2, rows[0].shape[0]*0.2))
    column = max([len(l) for l in fullCharacters])
    row = len(fullCharacters)

    subNum = 1
    for i, r in enumerate(fullCharacters):
        for j, c in enumerate(r):
            fig.add_subplot(row, column, subNum)
            subNum += 1
            plt.axis('off')
            plt.imshow(c)

    fig.tight_layout()

    # Comment to here if you don't need the plot



    # File Output

    current_directory = os.getcwd()

    final_directory = os.path.join(current_directory, 'Output')
    if os.path.exists(final_directory):
        shutil.rmtree(final_directory)

    os.makedirs(final_directory)

    row_directory = os.path.join(final_directory, 'Rows')
    os.makedirs(row_directory)

    for i, row_image in enumerate(rows):
        cv2.imwrite(os.path.join(row_directory, str(i)+'.png'), row_image)

    for i, row in enumerate(fullCharacters):
        i_dir = os.path.join(final_directory, str(i))
        os.makedirs(i_dir)
        for j, char_image in enumerate(row):
            cv2.imwrite(os.path.join(i_dir, str(j)+'.png'), char_image)


    return fullCharacters


# Run the program

def segmentationRun(input):

    img = cv2.imread(str(input), 0)
    cv2.imshow('img', img)
    output = seg(img)

    plt.show()
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# Run this program by segmentation.py [image_filename]

if len(sys.argv) >1:
    read_image = sys.argv[1]
    segmentationRun(read_image)
