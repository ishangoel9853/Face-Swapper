# --------------------------Importing libraries-------------------------

import cv2
import dlib
import numpy as np
import utilities as util


# --------------------------Image import-------------------------
img = cv2.imread('inp_Img/gadot.jpg')

l_mark = util.findLandmarks(img)
plotted_image = util.plotLandmarks(img, l_mark)
# cv2.imshow("Facial Landmarks", plotted_image)
cv2.imwrite("landmark.jpg", plotted_image)

cv2.waitKey(0)
cv2.destroyAllWindows()					
