import cv2
import numpy as np

# define names of each possible ArUco tag OpenCV supports
ARUCO_DICT = {
	"DICT_4X4_50": cv2.aruco.DICT_4X4_50,
	"DICT_4X4_100": cv2.aruco.DICT_4X4_100,
	"DICT_4X4_250": cv2.aruco.DICT_4X4_250,
	"DICT_4X4_1000": cv2.aruco.DICT_4X4_1000,
	"DICT_5X5_50": cv2.aruco.DICT_5X5_50,
	"DICT_5X5_100": cv2.aruco.DICT_5X5_100,
	"DICT_5X5_250": cv2.aruco.DICT_5X5_250,
	"DICT_5X5_1000": cv2.aruco.DICT_5X5_1000,
	"DICT_6X6_50": cv2.aruco.DICT_6X6_50,
	"DICT_6X6_100": cv2.aruco.DICT_6X6_100,
	"DICT_6X6_250": cv2.aruco.DICT_6X6_250,
	"DICT_6X6_1000": cv2.aruco.DICT_6X6_1000,
	"DICT_7X7_50": cv2.aruco.DICT_7X7_50,
	"DICT_7X7_100": cv2.aruco.DICT_7X7_100,
	"DICT_7X7_250": cv2.aruco.DICT_7X7_250,
	"DICT_7X7_1000": cv2.aruco.DICT_7X7_1000,
	"DICT_ARUCO_ORIGINAL": cv2.aruco.DICT_ARUCO_ORIGINAL,
	"DICT_APRILTAG_16h5": cv2.aruco.DICT_APRILTAG_16h5,
	"DICT_APRILTAG_25h9": cv2.aruco.DICT_APRILTAG_25h9,
	"DICT_APRILTAG_36h10": cv2.aruco.DICT_APRILTAG_36h10,
	"DICT_APRILTAG_36h11": cv2.aruco.DICT_APRILTAG_36h11
}

# generate a set of aruco marker images with desired id's, size, and border
def GenerateArucoMarkerImages(arucoDict,idSet,size,border):
	# get number of markers from id list
	N = idSet.shape[0]

	# initialize set of output images
	imgSet = np.zeros((size,size,1,N),dtype="uint8")

	# initialize counter
	k = 0

	# for each marker id
	for id_k in idSet:
		img = np.zeros((size,size,1),dtype="uint8")
		cv2.aruco.drawMarker(arucoDict,id_k,size,img,border)
		imgSet[:,:,:,k] = img
		k += 1

	return imgSet

def main():
	# setup tag dictionary object
	arucoDict = cv2.aruco.Dictionary_get(ARUCO_DICT["DICT_4X4_250"])
	
	# set marker parameters for generation
	size = 300
	border = 1
	# idSet = np.array([0,1,2,3,4,5,6,7,8,9])
	idSet = np.arange(0,100)
	N = idSet.shape[0]

	# generate marker images
	imgSet = GenerateArucoMarkerImages(arucoDict,idSet,size,border)

	# save marker images
	for k in range(0,N):
		img = imgSet[:,:,:,k]
		fname = "aruco" + str(k) + ".png"
		cv2.imwrite(fname,img)

if __name__ == '__main__':
	main()

