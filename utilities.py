# --------------------------Importing libraries-------------------------

import cv2
import dlib
import numpy as np


# --------------------------Defining Facial Points-------------------------

JAW_PTS = list(range(0, 17))
NOSE_PTS = list(range(27, 35))
FACE_PTS = list(range(17, 68))
MOUTH_PTS = list(range(48, 61))
RIGHT_BROW_PTS = list(range(17, 22))
LEFT_BROW_PTS = list(range(22, 27))
RIGHT_EYE_PTS = list(range(36, 42))
LEFT_EYE_PTS = list(range(42, 48))

LIST_1 = (LEFT_EYE_PTS+RIGHT_EYE_PTS+LEFT_BROW_PTS+RIGHT_BROW_PTS+MOUTH_PTS+NOSE_PTS)
LIST_2 = (LEFT_EYE_PTS+RIGHT_EYE_PTS+LEFT_BROW_PTS+RIGHT_BROW_PTS+NOSE_PTS+MOUTH_PTS)

# Path to shape predictor file for landmark detection
PATH = 'shape_predictor_68_face_landmarks.dat'

# Dlib predictor and detector objects
predictor = dlib.shape_predictor(PATH)
detector = dlib.get_frontal_face_detector()


# --------------------------Function Definitions-------------------------

def findLandmarks(image):
	'''
	FUNCTION : To detect the facial landmarks in an image using the dat file.
	INPUT : Image for facial Landmark detection.
	RETURNS : A 68x2 element matrix, each row of which corresponding with the x,y coordinates of a particular feature point in image.
	'''
	faces = detector(image, 1)

	if len(faces) == 0:
		return 'none'
	if len(faces) > 1:
		return 'multiple'

	lst = np.matrix([[ldmrk.x, ldmrk.y] for ldmrk in predictor(image, faces[0]).parts()])
	
	return lst


def plotLandmarks(image, landmarks):
	'''
	FUNCTION : To mark the given landmarks in the given image.
	INPUT : 'image' for facial Landmark marking, 
			'landmarks'- a 68x2 element matrix containing facial landmarks to be plotted.
	RETURNS : Image with the same resolution as the input image and the landmarks(given as input) highlighted.
	'''
	img = image.copy()
	
	for k, point in enumerate(landmarks):
		position = (point[0,0], point[0,1])
		cv2.putText(img, str(k), (position), fontFace=cv2.FONT_ITALIC, fontScale=0.4, color=(0,0,0))
		cv2.circle(img, position, 3, color=(0,255,0))

	return img


def convex_hull(image, points, color):
	'''
	FUNCTION : Computing the convex hull of the facial landmark points.
	INPUT : 'image' matrix, 
			'points'- a 68x2 element matrix containing facial landmarks, 'color' for filling the polygon.
	RETURNS : -
	'''
	points = cv2.convexHull(points)
	cv2.fillConvexPoly(image, points, color=color)


def getMask(image, landmarks):
	'''
	FUNCTION : To generate a mask for the image and the facial landmarks.
	INPUT : 'image' matrix, 
			'landmarks'- a 68x2 matrix containging facial landmark points for the 'image'.
	RETURNS : Image with the same resolution as the input image and the landmarks(given as input) highlighted.
	'''
	image = np.zeros(image.shape[:2], dtype=np.float64)

	for grp in LIST_2:
		convex_hull(image, landmarks[grp], color=1)

	image = np.array([image, image, image]).transpose((1,2,0))
	
	image = (cv2.GaussianBlur(image, (11,11), 0) > 0) * 1.0		
	image = cv2.GaussianBlur(image, (11,11), 0)

	return image


def transformation(pts1, pts2):
	'''
	FUNCTION : Calculate the degree of rotation between images(using Singlar Value Decomposition).
	INPUT : 'pts1' and 'pts2', two 68x2 matrices containing facial landmark points for image1 and image2 respectively.
	RETURNS : Complete transformation as an affine transformation matrix.
	'''
	pts1 = pts1.astype(np.float64)
	pts2 = pts2.astype(np.float64)

	mean_1 = np.mean(pts1, axis=0)
	mean_2 = np.mean(pts2, axis=0) 
	pts1 = pts1 - mean_1
	pts2 = pts2 - mean_2

	std_1 = np.std(pts1)
	std_2 = np.std(pts2)
	pts1 /= std_1
	pts2 /= std_2

	U, S, V = np.linalg.svd(pts1.T * pts2)
	R = (U * V).T
	trans = np.vstack([np.hstack(((std_2/std_1)*R, mean_2.T - (std_2/std_1) * R * mean_1.T)), np.matrix([0., 0., 1.])]) 

	return trans


def matchColor(image1, image2, landmarks, blur_factor=0.6):
	''''
	FUNCTION : Change the color of image2 to match the color of image1.
	INPUT : 'image1' and 'image2'- the input image matrices,
			'landmarks'- a 68x2 matrix with facial landmark points for image1, 
			'blur factor'- factor required for Gaussian blurring.
	RETURNS : image2 with the color similar to the color of image1.
	'''
	kernel = blur_factor * np.linalg.norm(np.mean(landmarks[LEFT_EYE_PTS], axis=0) - np.mean(landmarks[RIGHT_EYE_PTS], axis=0))
	kernel = int(kernel)

	if kernel % 2 == 0:
		kernel += 1

	img1_blur = cv2.GaussianBlur(image1, (kernel, kernel), 0)
	img2_blur = cv2.GaussianBlur(image2, (kernel, kernel), 0)

	img2_blur += (128 * (img1_blur <= 1.0)).astype(img2_blur.dtype)
	# print(image2_blur)
	# print(128 * (image1_blur <= 1.0))
	combined_img = image2.astype(np.float64) * img1_blur.astype(np.float64) / img2_blur.astype(np.float64)
	return combined_img



def warp_image(image, trans, shape):
	'''
	FUNCTION : Maps image2 to image1.
	INPUT : 'image' that needs to be warped,
			'trans'- affine transformation matrix
			'shape'- shape of image to which 'image' has to be warped.
	RETURNS : Warped image with the shape specified.
	'''
	warped = np.zeros(shape, dtype=image.dtype)
	cv2.warpAffine(image, trans[:2], (shape[1], shape[0]), flags=cv2.WARP_INVERSE_MAP, borderMode=cv2.BORDER_TRANSPARENT, dst=warped)

	return warped
