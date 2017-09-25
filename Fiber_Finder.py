# -*- coding: utf-8 -*-
"""
Created on Mon Sep 25 08:30:34 2017

@author: mkeranen
"""
#Compilation of following two tutorials:
#    http://www.pyimagesearch.com/2015/11/02/watershed-opencv/ &
#    http://www.learnopencv.com/filling-holes-in-an-image-using-opencv-python-c/

# import the necessary packages
from skimage.feature import peak_local_max
from skimage.morphology import watershed
from scipy import ndimage
import numpy as np
import cv2


# load the image and perform pyramid mean shift filtering
# to aid the thresholding step
image = cv2.imread('fibers.jpg');
image = image.copy()

#Remove centers of fibers
#____

# Threshold.
 
th, im_th = cv2.threshold(image, 190, 255, cv2.THRESH_BINARY);
 
# Copy the thresholded image.
im_floodfill = im_th.copy()
 
# Mask used to flood filling.
# Notice the size needs to be 2 pixels than the image.
h, w = im_th.shape[:2]
mask = np.zeros((h+2, w+2), np.uint8)
 
# Floodfill from point (0, 0)
cv2.floodFill(im_floodfill, mask, (0,0), 255);
 
# Invert floodfilled image
im_floodfill_inv = cv2.bitwise_not(im_floodfill)
 
# Combine the two images to get the foreground.
im_out = im_th | im_floodfill_inv
 
#___
# convert the mean shift image to grayscale, then apply
# Otsu's thresholding
gray = cv2.cvtColor(im_out, cv2.COLOR_BGR2GRAY)
thresh = cv2.threshold(gray, 0, 255,cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
#cv2.imshow("Thresh", thresh)

# compute the exact Euclidean distance from every binary
# pixel to the nearest zero pixel, then find peaks in this
# distance map
D = ndimage.distance_transform_edt(thresh)
localMax = peak_local_max(D, indices=False, min_distance=10,
	labels=thresh)

# perform a connected component analysis on the local peaks,
# using 8-connectivity, then appy the Watershed algorithm
markers = ndimage.label(localMax, structure=np.ones((3, 3)))[0]
labels = watershed(-D, markers, mask=thresh)
print("[INFO] {} unique segments found".format(len(np.unique(labels)) - 1))

# loop over the unique labels returned by the Watershed
# algorithm
for label in np.unique(labels):
    # if the label is zero, we are examining the 'background'
    # so simply ignore it
    if label == 0:
    	continue
    # otherwise, allocate memory for the label region and draw
    # it on the mask
    mask = np.zeros(gray.shape, dtype="uint8")
    mask[labels == label] = 255
    
    # detect contours in the mask and grab the largest one
    cnts = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL,
    	cv2.CHAIN_APPROX_SIMPLE)[-2]
    c = max(cnts, key=cv2.contourArea)
    
    #Bounding Rect
    x,y,w,h = cv2.boundingRect(c)
    if w < 32 and h < 32:
        cv2.rectangle(image, (x,y), (x+w, y+h),(0,255,0),1)
        cv2.putText(image, "{}".format(label), (int(x), int(y)+int(h/1.5)),
    	 cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)
        cv2.imwrite('found_fibers.jpg', image)
        newImg = image[y:y+h, x:x+h]
        newFileName = 'ExtractedFibers\Fiber{}.jpg'.format(label)
        cv2.imwrite(newFileName, newImg)
        

    ## draw a circle enclosing the object
    #((x, y), r) = cv2.minEnclosingCircle(c)
    #cv2.circle(image, (int(x), int(y)), int(r), (0, 255, 0), 2)
    #cv2.putText(image, "#{}".format(label), (int(x) - 10, int(y)),
    #	cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

# show the output image
cv2.imshow("Output", image)
cv2.waitKey(0)