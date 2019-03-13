import cv2 
from PIL import Image
import os
import numpy as np

def crop(image):
        cv_img = cv2.CreateImageHeader(image.size, cv2.IPL_DEPTH_8U, 3)  # RGB image
        cv.SetData(cv_img, pil_img.tostring(), pil_img.size[0]*3)
        
        gray = cv2.cvtColor(cv_img, cv2.COLOR_BGR2GRAY)
        gray = cv2.blur(gray, (10, 10))
        ret, thresh = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY)
        contours, hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        c = max(contours, key=cv2.contourArea)

        # DRAW
        # drawing = image
        # cv2.drawContours(drawing, c, -1, (0, 0, 255), 2)
        # cv2.imshow("draw", drawing)

        x, y, w, h = cv2.boundingRect(c)
        # Padding
        dx = 10
        dy = 10
        cv_img = cv_img[y-dy:y+h+dy, x+dx:x+w+dx]
        
        return Image.fromstring("L", cv.GetSize(cv_img), cv_img.tostring())

if __name__ == '__main__':
    path = "../corrected-boneage-training-dataset/"

    for file in os.listdir(path):
        image = cv2.imread(path + file)
        image = cv2.resize(image, (0, 0), None, 0.5, 0.5)
        
        image = crop(image)
        
        # SAVE
        #cv2.imwrite("../contour-boneage-training-dataset/" + file, new_img)

        # COMPARE
        cv2.imshow("before", image)
        cv2.imshow("after", new_img)

        cv2.waitKey(0)

    cv2.destroyAllWindows()