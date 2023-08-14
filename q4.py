import numpy as np

import skimage
import skimage.measure
import skimage.color
import skimage.restoration
import skimage.filters
import skimage.morphology
import skimage.segmentation
import matplotlib.pyplot as plt

# takes a color image
# returns a list of bounding boxes and black_and_white image

# insert processing in here
# one idea estimate noise -> denoise -> greyscale -> threshold -> morphology -> label -> skip small boxes 
# this can be 10 to 15 lines of code using skimage functions

##########################
##### your code here #####
##########################
def findLetters(image):
    bboxes = []
    bw = None
  
    greyscale = skimage.color.rgb2gray(image)
    image = skimage.filters.gaussian(greyscale,3)
    thresh = skimage.filters.threshold_otsu(greyscale)
    
    bw = greyscale > thresh

    morph = skimage.morphology.opening(bw, np.ones((9,9))) #, np.ones((6,6))
    morph = skimage.morphology.erosion(morph, np.ones((3,3))) #, np.ones((,7))
  
    labels = skimage.measure.label(1 - morph)
    
    for region in skimage.measure.regionprops(labels):
        if region.area >= 70:
            #Increasing the size of the bounding box
            margin = 15
            min1, min2, max1, max2 = region.bbox
            out_bbx = [min1 - margin, min2- margin, max1 + margin, max2 + margin]
            if max1 - min1 >= 30 and max2 - min2 >= 10:
                bboxes.append(out_bbx)

    bw = morph

    return bboxes, bw

# plt.figure()
# plt.imshow(morph)
# plt.show()
# plt.figure()
# plt.imshow(bw)
    