# --------------------------Importing libraries-------------------------

import cv2
import dlib
import numpy as np
import utilities as util


# --------------------------Scaling Function-------------------------

def read_features(image):
	img = image
	img = cv2.resize(img, None, fx=0.75, fy=0.75, interpolation=cv2.INTER_LINEAR)
	img = cv2.resize(img, (img.shape[1]* 1, img.shape[0]*1))

	landmarks = util.get_landmarks(img)

	return img, landmarks


# --------------------------Driver Function-------------------------

def swapped(image1 , image2):
	'''
	Combines all function and outputs a swapped image
	'''
	# image1, landmarks1 = image1, check
	image1, landmarks1 = read_features(image1)
	image2, landmarks2 = read_features(image2)

	if landmarks1 == 'multiple':
		print ('ERROR : Multiple faces detected.')
		return image1

	if landmarks1 == 'none':
		print ('ERROR : Face not detected.')
		return image1

	# image1, landmarks1 = image1, check
	# image2, landmarks2 = read_features(image2)	

	M = util.transform_points(landmarks1[util.ALIGN_POINTS], landmarks2[util.ALIGN_POINTS])
	
	mask = util.face_mask(image2, landmarks2)
	warped_mask = util.warp_image(mask, M, image1.shape)
	combined_mask = np.max([util.face_mask(image1, landmarks1), warped_mask], axis=0)

	warped_image2 = util.warp_image(image2, M, image1.shape)
	warped_image2_new = util.mix_colors(image1, warped_image2, landmarks1)

	final_output = image1 * (1.0 - combined_mask) + warped_image2_new * combined_mask
	cv2.imwrite("SwappedImage.jpg", final_output)
	o = cv2.imread("SwappedImage.jpg")
	return o


# --------------------------Image import-------------------------

capture = cv2.VideoCapture(0)
image = cv2.imread('inp_Img/Trump.jpg')
use_dlib = False

while True:
	response, frame = capture.read()
	frame = cv2.resize(frame, None, fx=0.8, fy=0.8, interpolation=cv2.INTER_LINEAR)
	frame = cv2.flip(frame, 1)
	cv2.imshow("YOU LOOK LIKE", swapped(frame, image))
	if cv2.waitKey(1) & 0xFF == ord('q'):
		break

capture.release()
cv2.destroyAllWindows()








