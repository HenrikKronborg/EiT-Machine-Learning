import cv2
import os
import numpy as np

if __name__ == '__main__':
    path = "../corrected-boneage-training-dataset/"

    for file in os.listdir(path):
        image = cv2.imread(path + file)

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        gray = cv2.blur(gray, (10, 10))

        ret, thresh = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY)

        contours, hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        drawing = image

        c = max(contours, key=cv2.contourArea)

        #cv2.drawContours(drawing, c, -1, (0, 0, 255), 2)

        x, y, w, h = cv2.boundingRect(c)
        new_img = image[y:y + h, x:x + w]

        cv2.imwrite("../contour-boneage-training-dataset/" + file, new_img)

        # Legg ved for å teste i stedet for å lagre
        '''
        cv2.imshow('image', new_img)
        cv2.waitKey(0)
        '''

    cv2.destroyAllWindows()