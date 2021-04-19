import numpy as np
import cv2
import matplotlib.pyplot as plt
import os
import shutil
import sys
import math



# Threshold for Row Segmentation
# Make it large if row's height too short
# Range 0.9990 to 0.9999
tsRow = 0.9999

# Threshold for Character Segmentation
# Make it large if character not fully displayed
# Range 0.93 to 0.98
tsChar = 0.94

# Whether output the segmented images to the file
output_file = True

# Whether show the plot of all segmented images
plot = False


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

    output = input.copy()

    for rota in range(4):
        output = np.rot90(output)
        height, width = output.shape

        white = 0
        edge = []

        point = []
        for i in range(height):
            sum = 0
            lastwhite = white
            for j in range(width):
                sum += output[i, j]

            if(sum < width*255*threshold):
                edge.append(i)

        output = output[edge[0]: height]
        input = center(output, 50, 140)

    return input

def center(input, size, threshold):

    height, width = input.shape
    char_size = size*0.9
    ratio = char_size / max(height, width)

    if(int(width*ratio) < 2 or int(height*ratio) < 2):
        resized = cv2.resize(input, (2, 2))
    else:
        resized = cv2.resize(input, (int(width*ratio), int(height*ratio)))

    charHeight, charWidth = resized.shape

    panel = np.ones((size, size))*255

    for i in range(-math.floor(charHeight/2), math.floor(charHeight/2)):
        for j in range(-math.floor(charWidth/2), math.floor(charWidth/2)):
            if(resized[math.floor(charHeight/2) + i, math.floor(charWidth/2) + j] < threshold):
                panel[math.floor(size/2) + i, math.floor(size/2) + j] = 0

    return panel


def seg(input, thresholdRow = tsRow, thresholdChar = tsChar):

    rows = segRow(input, thresholdRow)

    fullCharacters = []
    for r in rows:
        fullCharacters.append(segChar(r, thresholdChar))


    # Plot generation

    if(plot):

        plotSegmented(fullCharacters, rows)



    # File Output

    if(output_file):

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

def plotSegmented(fullCharacters, rows):

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



# Run the program

def segmentationRun(input):

    img = cv2.imread(str(input), 0)
    cv2.imshow('img', img)
    output = seg(img)



    plt.show()
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# Run this program by segmentation.py [image_filename]

if __name__ == "__main__":

    if len(sys.argv) >1:
        read_image = sys.argv[1]
        plot = True
        if len(sys.argv) >2:
            output_file = False if sys.argv[2] == "false" else True
        segmentationRun(read_image)
