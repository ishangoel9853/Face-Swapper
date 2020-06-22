# --------------------------Importing libraries-------------------------

import cv2
import dlib
import numpy as np
import utilities as util


# --------------------------Scaling Function-------------------------

def scale_landmark(image):
	img = image
	img = cv2.resize(img, None, fx=0.75, fy=0.75, interpolation=cv2.INTER_LINEAR)
	landmarks = util.findLandmarks(img)
	return img, landmarks


# --------------------------Driver Function-------------------------

def base_func(image1 , image2):
	'''
	FUNCTION : A driver function that combines all the underlying functions.
	INPUT : 'image1'- the image matrix for the base image,
			'image2'- the image matrix for the image that has to be superimposed on image1.
	RETURNS : A final image with the faces swapped.
	'''
	# image1, ldmrk1 = image1, check
	image1, ldmrk1 = scale_landmark(image1)
	image2, ldmrk2 = scale_landmark(image2)

	if ldmrk1 == 'multiple':
		print ('ERROR : Multiple faces detected.')
		return image1

	if ldmrk1 == 'none':
		print ('ERROR : Face not detected.')
		return image1

	# image1, ldmrk1 = image1, check
	# image2, ldmrk2 = scale_landmark(image2)	

	M = util.transformation(ldmrk1[util.LIST_1], ldmrk2[util.LIST_1])
	
	mask = util.getMask(image2, ldmrk2)
	warped_mask = util.warp_image(mask, M, image1.shape)
	combined_mask = np.max([util.getMask(image1, ldmrk1), warped_mask], axis=0)

	warped_img2 = util.warp_image(image2, M, image1.shape)
	colorMatched_img2 = util.matchColor(image1, warped_img2, ldmrk1)

	swapped_img = image1 * (1.0 - combined_mask) + colorMatched_img2 * combined_mask
	cv2.imwrite("liveSwappedImage.jpg", swapped_img)
	saved = cv2.imread("liveSwappedImage.jpg")
	return saved


# --------------------------Image import-------------------------

camera = cv2.VideoCapture(0)
img = cv2.imread('inp_Img/Trump.jpg')

while True:
	response, frame = camera.read()
	frame = cv2.resize(frame, None, fx=0.8, fy=0.8, interpolation=cv2.INTER_LINEAR)
	frame = cv2.flip(frame, 1)
	
	cv2.imshow("Live Face Swap", base_func(frame, img))
	if cv2.waitKey(1) & 0xFF == ord('q'):
		break

camera.release()
cv2.destroyAllWindows()








