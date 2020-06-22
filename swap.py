# --------------------------Importing libraries-------------------------

import cv2
import dlib
import numpy as np
import utilities as util


# --------------------------Driver Function-------------------------

def base_func(image1 , image2):
	'''
	FUNCTION : A driver function that combines all the underlying functions.
	INPUT : 'image1'- the image matrix for the base image,
			'image2'- the image matrix for the image that has to be superimposed on image1.
	RETURNS : A final image with the faces swapped.
	'''
	ldmrk1 = util.findLandmarks(image1)
	ldmrk2 = util.findLandmarks(image2)	

	# if ldmrk1 == 'multiple':
	# 	print ('ERROR : Multiple faces detected.')
	# 	return image1

	# if ldmrk1 == 'none':
	# 	print ('ERROR : Face not detected.')
	# 	return image1

	M = util.transformation(ldmrk1[util.LIST_1], ldmrk2[util.LIST_1])
	
	mask = util.getMask(image2, ldmrk2)
	warped_mask = util.warp_image(mask, M, image1.shape)
	combined_mask = np.max([util.getMask(image1, ldmrk1), warped_mask], axis=0)

	warped_img2 = util.warp_image(image2, M, image1.shape)
	colorMatched_img2 = util.matchColor(image1, warped_img2, ldmrk1)

	swapped_img = image1 * (1.0 - combined_mask) + colorMatched_img2 * combined_mask
	
	return swapped_img


# --------------------------Image import-------------------------

img1 = cv2.imread('inp_Img/Trump.jpg')
img2 = cv2.imread('inp_Img/jongun.jpg')
# print(img1.shape)

# face of image2 swapped on image1
swapped = base_func(img1, img2)
# cv2.imshow("Swapped image", swapped)	
cv2.imwrite("Swapped_Image.jpg", swapped)


cv2.waitKey(0)
cv2.destroyAllWindows()








