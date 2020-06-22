# --------------------------Importing libraries-------------------------

import cv2
import dlib
import numpy as np
import utilities as util


# --------------------------Driver Function-------------------------

def swapped(image1 , image2):
	'''
	FUNCTION : A driver function that combines all the underlying functions.
	INPUT : 'image1'- the image matrix for the base image,
			'image2'- the image matrix for the image that has to be superimposed on image1.
	RETURNS : A final image with the faces swapped.
	'''
	landmarks1 = util.get_landmarks(image1)
	landmarks2 = util.get_landmarks(image2)	

	M = util.transform_points(landmarks1[util.ALIGN_POINTS], landmarks2[util.ALIGN_POINTS])
	
	mask = util.face_mask(image2, landmarks2)
	warped_mask = util.warp_image(mask, M, image1.shape)
	combined_mask = np.max([util.face_mask(image1, landmarks1), warped_mask], axis=0)

	warped_image2 = util.warp_image(image2, M, image1.shape)
	warped_image2_new = util.mix_colors(image1, warped_image2, landmarks1)

	final_output = image1 * (1.0 - combined_mask) + warped_image2_new * combined_mask
	cv2.imwrite("SwappedImage3.jpg", final_output)
	return final_output


# --------------------------Image import-------------------------

image1 = cv2.imread('inp_Img/Trump.jpg')
image2 = cv2.imread('inp_Img/jongun.jpg')

# face of image2 swapped on image1
swapped_image = swapped(image1, image2)
#cv2.imshow("Swapped 1", swapped_image)	


cv2.waitKey(0)
cv2.destroyAllWindows()








