import cv2
import dlib
import numpy as np
from utilities import get_landmarks
from utilities import mark_landmarks

# Reading our image
img = cv2.imread('inp_Img/Hillary.jpg')


l = get_landmarks(img)
marked_image = mark_landmarks(img, l)

cv2.imshow("Landmarks", marked_image)
cv2.waitKey(0)
cv2.destroyAllWindows()					
