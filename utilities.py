import cv2
import dlib
import numpy as np



JAW_POINTS = list(range(0, 17))
NOSE_POINTS = list(range(27, 35))
FACE_POINTS = list(range(17, 68))
MOUTH_POINTS = list(range(48, 61))
RIGHT_BROW_POINTS = list(range(17, 22))
LEFT_BROW_POINTS = list(range(22, 27))
RIGHT_EYE_POINTS = list(range(36, 42))
LEFT_EYE_POINTS = list(range(42, 48))

ALIGN_POINTS = (LEFT_EYE_POINTS+RIGHT_EYE_POINTS+LEFT_BROW_POINTS+RIGHT_BROW_POINTS+MOUTH_POINTS+NOSE_POINTS)

OVERLAY_POINTS = (LEFT_EYE_POINTS+RIGHT_EYE_POINTS+LEFT_BROW_POINTS+RIGHT_BROW_POINTS+NOSE_POINTS+MOUTH_POINTS)

# Path to shape predictor file
PATH = 'shape_predictor_68_face_landmarks.dat'

# Our landpoints' predictor and detector objects
predictor = dlib.shape_predictor(PATH)
detector = dlib.get_frontal_face_detector()

def get_landmarks(image):
	'''
	Returns a 68x2 element matrix, each row of which corresponding with the
	x, y coordinates of a particular feature point in image.
	'''
	points = detector(image, 1)

	if len(points) > 1:
		return 'error'
	if len(points) == 0:
		return 'error'

	return np.matrix([[t.x, t.y] for t in predictor(image, points[0]).parts()])

def mark_landmarks(image, landmarks):
	image = image.copy()
	for i, point in enumerate(landmarks):
		position = (point[0,0], point[0,1])
		cv2.putText(image, str(i), (position), fontFace=cv2.FONT_ITALIC, fontScale=0.4, color=(0,0,0))
		cv2.circle(image, position, 3, color=(0,255,0))

	return image


def convex_hull(image, points, color):
	points = cv2.convexHull(points)
	cv2.fillConvexPoly(image, points, color=color)



def face_mask(image, landmarks):
	'''
	Generate a mask for the image and a landmark matrix
	'''
	image = np.zeros(image.shape[:2], dtype=np.float64)

	for grp in OVERLAY_POINTS:
		convex_hull(image, landmarks[grp], color=1)

	image = np.array([image, image, image]).transpose((1,2,0))
	image = (cv2.GaussianBlur(image, (11,11), 0) > 0) * 1.0		
	image = cv2.GaussianBlur(image, (11,11), 0)

	return image



def transform_points(p1, p2):
	'''
	Calculates the rotation portion using the Singular Value Decomposition and
	Return the complete transformaton as an affine transformation matrix.
	'''
	p1 = p1.astype(np.float64)
	p2 = p2.astype(np.float64)

	t1 = np.mean(p1, axis=0)
	t2 = np.mean(p2, axis=0)
	p1 -= t1
	p2 -= t2

	s1 = np.std(p1)
	s2 = np.std(p2)
	p1 /= s1
	p2 /= s2

	U, S, V = np.linalg.svd(p1.T * p2)
	R = (U * V).T

	return np.vstack([np.hstack(((s2/s1)*R, t2.T - (s2/s1) * R * t1.T)), np.matrix([0., 0., 1.])])




def mix_colors(image1, image2, landmarks, blur_factor=0.6):
	'''
	Changes the colouring of image2 to match that of image1
	'''

	blurred = blur_factor * np.linalg.norm(np.mean(landmarks[LEFT_EYE_POINTS], axis=0) - np.mean(landmarks[RIGHT_EYE_POINTS], axis=0))
	blurred = int(blurred)

	if blurred % 2 == 0:
		blurred += 1
	image1_blur = cv2.GaussianBlur(image1, (blurred, blurred), 0)
	image2_blur = cv2.GaussianBlur(image2, (blurred, blurred), 0)

	image2_blur += (128 * (image1_blur <= 1.0)).astype(image2_blur.dtype)

	return (image2.astype(np.float64) * image1_blur.astype(np.float64) / image2_blur.astype(np.float64))




def warp_image(image, M, shape):
	'''
	Maps the second image onto the first and return ithe same
	'''
	initial = np.zeros(shape, dtype=image.dtype)
	cv2.warpAffine(image, M[:2], (shape[1], shape[0]), dst=initial, borderMode=cv2.BORDER_TRANSPARENT, flags=cv2.WARP_INVERSE_MAP)

	return initial
