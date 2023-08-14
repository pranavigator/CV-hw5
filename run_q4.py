import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches

import skimage
import skimage.measure
import skimage.color
import skimage.restoration
import skimage.io
import skimage.filters
import skimage.morphology
import skimage.segmentation

from nn import *
from q4 import *
# do not include any more libraries here!
# no opencv, no sklearn, etc!
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=UserWarning)

def GetCorrectAnswer(image_name):
    if image_name == "01_list.jpg":
        return ["TODOLIST",
                "1MAKEATODOLIST",
                "2CHECKOFFTHEFIRST",
                "THINGONTODOLIST",
                "3REALIZEYOUHAVEALREADY",
                "COMPLETED2THINGS",
                "4REWARDYOURSELFWITH",
                "ANAP"]
    if image_name == "02_letters.jpg":
        return ["ABCDEFG",
                "HIJKLMN",
                "OPQRSTU",
                "VWXYZ",
                "1234567890"]
    if image_name == "03_haiku.jpg":
        return ["HAIKUSAREEASY",
                "BUTSOMETIMESTHEYDONTMAKESENSE",
                "REFRIGERATOR"]
    if image_name == "04_deep.jpg":
        return ["DEEPLEARNING",
                "DEEPERLEARNING",
                "DEEPESTLEARNING"]

total_letter_count = 0
total_match_count = 0
for img in os.listdir('../images'):
    im1 = skimage.img_as_float(skimage.io.imread(os.path.join('../images',img)))
    bboxes, bw = findLetters(im1)
    
    # find the rows using..RANSAC, counting, clustering, etc.
    ##########################
    ##### your code here #####
    ##########################
    rows = []
    row = []

    #Top row stays the same as the first element in the bboxes list
    row_top_center = (bboxes[0][2] + bboxes[0][0]) // 2

    row.append(bboxes[0])

    for i in range(1, len(bboxes)):
        current_row_min = bboxes[i][0]
        current_row_max = bboxes[i][2]
        if row_top_center >= (current_row_min - 70) and row_top_center <= (current_row_max + 70):
            row.append(bboxes[i])
        else:
            #Creating a new center
            rows.append(row)
            row = []
            row_top_center = (bboxes[i][2] + bboxes[i][0]) // 2
            row.append(bboxes[i])
    rows.append(row)

    #Reordering rows because the letters are out of order

    for row_idx in range(len(rows)):
        row = rows[row_idx]
        for i in range(len(row)):
            for j in range(i,len(row)):
                if rows[row_idx][j][1] < rows[row_idx][i][1]:
                   tmp_row = rows[row_idx][i]
                   rows[row_idx][i] = rows[row_idx][j]
                   rows[row_idx][j] = tmp_row


    # print("New image:\n")
    # for i in range(len(rows)):
        
    #     print(len(rows[i]))

    # crop the bounding boxes
    # note.. before you flatten, transpose the image (that's how the dataset is!)
    # consider doing a square crop, and even using np.pad() to get your images looking more like the dataset
    ##########################
    ##### your code here #####
    ##########################
    
    crop_list = []
    row_crop = []

    plt.imshow(bw)
    for bbox in bboxes:
        minr, minc, maxr, maxc = bbox
        rect = matplotlib.patches.Rectangle((minc, minr), maxc - minc, maxr - minr,
                                fill=False, edgecolor='red', linewidth=2)
        plt.gca().add_patch(rect)
    plt.show()

    for row in rows:
        # print("new row")
        # print(row)
        
        for j in range(len(row)):
            min1, min2, max1, max2 = row[j]
            # print(min1)
            # print(row[j])

            #cropping character out
            crop = bw[min1:max1, min2:max2]
            
            #Square cropping
            crop_new = np.ones((np.max(crop.shape), np.max(crop.shape)))
            crop_center = crop_new.shape[0] // 2

            crop_new[crop_center - crop.shape[0]//2:crop_center + (crop.shape[0]-crop.shape[0]//2), 
                     crop_center - crop.shape[1]//2:crop_center + (crop.shape[1]-crop.shape[1]//2)] = crop


            crop = skimage.transform.resize(crop_new, (32,32))
            crop = skimage.morphology.erosion(crop)
            # crop = skimage.morphology.dilation(crop)
            # plt.figure()
            # plt.imshow(crop)
            # plt.show()
            crop = np.transpose(crop)
            row_crop.append(crop.flatten())
        crop_list.append(row_crop)


    # load the weights
    # run the crops through your neural network and print them out
    import pickle
    import string
    letters = np.array([_ for _ in string.ascii_uppercase[:26]] + [str(_) for _ in range(10)])
    # print(letters)
    params = pickle.load(open('q3_weights.pickle','rb'))
    ##########################
    ##### your code here #####
    ##########################
    for row_crop in crop_list:
        row_text = ""
        h1 = forward(row_crop, params, 'layer1')
        probs = forward(h1, params, 'output', softmax)
        preds = np.argmax(probs,axis=1)

        # print(preds)
        for k in range(preds.size):
            row_text += letters[preds[k]]

    #Calculating accuracy
    print(row_text + "\n")

    label = GetCorrectAnswer(img)

    #Flattening list
    label_flat = []
    for row in label:
        for letter in row:
            # print(element)
            label_flat.append(letter)
    
    total_letter_count += len(label_flat)
    match_count = 0
    for l in range(len(label_flat)):
        if row_text[l] == label_flat[l]:
            match_count += 1
    
    total_match_count += match_count

    accuracy = match_count / len(label_flat)
    print("Row Accuracy:", accuracy)

total_accuracy = total_match_count/total_letter_count
print("\nTotal Accuracy:", total_accuracy)


    
    