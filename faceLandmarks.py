import cv2
import dlib
import numpy as np
import utilities as util
# Reading our image
img = cv2.imread('inp_Img/Hillary.jpg')


l = util.get_landmarks(img)
marked_image = util.mark_landmarks(img, l)

cv2.imshow("Landmarks", marked_image)
cv2.waitKey(0)
cv2.destroyAllWindows()					
