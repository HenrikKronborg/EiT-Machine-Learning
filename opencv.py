import cv2
import numpy as np

path = "../boneage-test-dataset"

img = cv2.imread(path + "/1941.png")

#img = cv2.imread("cat.jpg")
img = cv2.resize(img, (500,600))

gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
#ret, thresh = cv2.threshold(gray,127,255,0)
ret, thresh = cv2.threshold(gray, 90, 255, cv2.THRESH_BINARY)

contours, hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

drawing = img
for i in range(len(contours)):
    cv2.drawContours(drawing, contours, i, (0,0,255), 2)

cv2.imshow('image', drawing)

cv2.waitKey(0)
cv2.destroyAllWindows()