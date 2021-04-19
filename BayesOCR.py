import numpy as np
import cv2
import os
import math
import pandas as pd
import sys


#from segmentation import *
import segmentation

pixel_difference = 10
scale = 5

output_file = True



def getAlpha(folder = "BayesDataset"):
    alphas = []

    print("Getting alpha dataset ..")
    if(not os.path.isdir(os.path.join(os.getcwd(), folder))):
        print("Cannot find dataset folder. ")
        exit()


    for filename in os.listdir(os.path.join(os.getcwd(), folder)):
        img = cv2.imread(os.path.join(os.getcwd(), folder, filename), 0)
        if img is not None:
            name = filename.replace('.png', '')
            sample = (str(name), img)
            alphas.append(sample)


    return alphas

def getSimilarTable(input, alphas):


    height, width = input.shape

    similarTable = []

    print("Generating similarity table .. ")


    for a in alphas:

        alpha_height, alpha_width = a[1].shape
        #print(alpha_height, alpha_width)

        scan_size = math.floor(alpha_height / scale)
        #print(scan_size)

        aimage = cv2.resize(a[1].copy(), (scan_size, scan_size))
        input = cv2.resize(input.copy(), (scan_size, scan_size))



        for i in range(scan_size):
            for j in range(scan_size):

                similar = 0

                diff = abs(int(input[i, j]) - int(aimage[i, j]))
                if(diff < pixel_difference):
                    similar = 1

                # ['alphabet', 'position', 'similarity']
                similarTable.append((a[0], (i, j), similar))

    # if(output_file):
    #     pd.DataFrame(similarTable).to_excel("Similarity Table.xlsx")

    return similarTable




def getProbTable(similarTable, dataset):


    print("Generating Prob Table..")
    print("Sample size: ", len(similarTable))
    probTable = []
    lastA = " "
    for similarity in similarTable:

        pixel = similarity[1]

        if(similarity[0] != lastA):
            print("Calculating prob for ", str(similarity[0])[0])
            lastA = similarity[0]


        sum=0
        for i in similarTable:
            if(i[1] == pixel and i[2] == 1):
                sum += 1


        if(sum == 0):
            probTable.append((similarity[0], similarity[1], 0))
        else:
            prob = (1/sum) if similarity[2] == 1 else (0.1/(len(dataset)+1))

            # ['alphabet', 'position', 'probability']
            probTable.append((similarity[0], similarity[1], prob))

    # if(output_file):
    #     pd.DataFrame(probTable).to_excel("Probability Table.xlsx")

    return probTable

def getFinalTable(probTable):

    finalTable = []
    checked = []

    for p in probTable:

        guess = p[0]

        if(guess not in checked):
            product = 1
            for q in probTable:
                if(q[0 not in checked]):
                    if(q[0] == guess):
                        if(round(q[2], 3) != 0):
                            product *= q[2]

                            if(product == 0):
                                print("Reach maximum of python digit, consider increase recognition scale")
                                break
            print("Computing Bayes Score: ", guess[0])

            checked.append(guess)

            # ['alphabet', 'image']
            finalTable.append((guess, product))

    finalTable.sort(key=lambda x: x[1], reverse=True)
    print("\nRead: ")
    print(finalTable[0][0][0], "-->", finalTable[0][1])
    print(finalTable[1][0][0], "-->", finalTable[0][1])
    print(finalTable[2][0][0], "-->", finalTable[0][1])

    return finalTable

def run(input_name, dataset_name = "BayesDataset"):
    img = cv2.imread(input_name, 0)
    segmented = segmentation.seg(img)
    dataset_images = getAlpha(dataset_name)

    for row in segmented:
        for char in row:
            similar_table = getSimilarTable(char, dataset_images)
            probability_table = getProbTable(similar_table, dataset_images)
            getFinalTable(probability_table)

            cv2.imshow("char", char)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

if __name__ == "__main__":

    if len(sys.argv) >1:
        read_image = sys.argv[1]
        segmentation.output_file = output_file
        if len(sys.argv) >2:
            dataset = sys.argv[2]
            run(read_image, dataset)
        else:
            run(read_image)
