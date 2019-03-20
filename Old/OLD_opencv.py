import cv2 
from PIL import Image
import os
import numpy as np

def crop(image):
        arr = np.asarray(image).copy()
        
        #gray = cv2.cvtColor(arr, cv2.COLOR_BGR2GRAY)
        gray = cv2.blur(arr, (10, 10))
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
        cv_img = arr[max(0, y-dy):y+h+dy, max(0, x+dx):x+w+dx]
        
        cv2.imshow("ttttt", cv_img)
        
        return Image.fromarray(cv_img)

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