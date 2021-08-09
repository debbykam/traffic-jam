import cv2
import numpy as np
import imutils

#load original image
img=cv2.imread(r'C:\Users\eitan\PycharmProjects\ImgVideoEdit\Smart Traffic Light\automobile-1835634_1280.jpg')
kernel= np.ones((5,5), np.uint8)

img = cv2.resize(img, (620,480) )
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) #convert to grey scale
gray = cv2.bilateralFilter(gray, 13, 15, 15) #filter to eliminate what we don't want
edged = cv2.Canny(gray, 30, 200) #Perform Edge detection
contours=cv2.findContours(edged.copy(),cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE) #find contours in our image
contours = imutils.grab_contours(contours) # separates the different shapes in our image
cnts = sorted(contours,key=cv2.contourArea, reverse = True)[:10] #organizes the shapes by size and takes the smallest 10 objects
screenCnt = None #counts

for c in cnts:
    # approximate the contour
    peri = cv2.arcLength(c, True)
    approx = cv2.approxPolyDP(c, 0.018 * peri, True)
    # if our approximated contour has four points, then
    # we can assume that we have found our screen
    if len(approx) == 4:
        screenCnt = approx
        break

# Masking the part other than the number plate
mask = np.zeros(gray.shape,np.uint8)
new_image = cv2.drawContours(img,mask[screenCnt],0,255,-1,)
new_image = cv2.bitwise_and(10,10,mask=mask)



#show image
cv2.imshow("Our car", edged)
cv2.waitKey(0) #o infinite delay