# USAGE
# python predict_batch.py --model output.h5 --labels logos/retinanet_classes.csv
#	--input logos/images --output output

# import the necessary packages
from keras_retinanet.utils.image import preprocess_image
from keras_retinanet.utils.image import read_image_bgr
from keras_retinanet.utils.image import resize_image
from collections import defaultdict as dd
from keras_retinanet import models
from skimage import img_as_ubyte
from PIL import Image
import numpy as np
import pandas as pd
import datetime
import argparse
import cv2
import os

#video settings \
from imutils.video import VideoStream
from imutils.video import FPS
import imutils
import time

#keras face detection setup
from keras.preprocessing.image import img_to_array
from keras.models import load_model
from keras.utils import get_file
import cvlib as cv

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-m", "--model", default = '../../resnet50_coco_best_v2.1.0.h5',
	help="path to pre-trained model")
ap.add_argument("-i", "--input", required=True,
	help="path to input video")
#ap.add_argument("-o", "--output", required=True,
#	help="path to directory to store output images")
ap.add_argument("-c", "--confidence", type=float, default=0.5,
	help="minimum probability to filter weak detections")
ap.add_argument("-o", "--output", required=True,
	help="path to output")
#ap.add_argument("-m", "--manneframes", type=int, default=50,
#	help="the number of stationery detections equals a mannequin")
ap.add_argument("-s", "--skipframes", type=int, default=5,
	help="the number frames to skip")
args = vars(ap.parse_args())

# load the class label mappings
LABELS = {0: 'person', 1: 'bicycle', 2: 'car', 3: 'motorcycle', 4: 'airplane', 5: 'bus', 6: 'train', 7: 'truck', 8: 'boat', 9: 'traffic light', 10: 'fire hydrant', 11: 'stop sign', 12: 'parking meter', 13: 'bench', 14: 'bird', 15: 'cat', 16: 'dog', 17: 'horse', 18: 'sheep', 19: 'cow', 20: 'elephant', 21: 'bear', 22: 'zebra', 23: 'giraffe', 24: 'backpack', 25: 'umbrella', 26: 'handbag', 27: 'tie', 28: 'suitcase', 29: 'frisbee', 30: 'skis', 31: 'snowboard', 32: 'sports ball', 33: 'kite', 34: 'baseball bat', 35: 'baseball glove', 36: 'skateboard', 37: 'surfboard', 38: 'tennis racket', 39: 'bottle', 40: 'wine glass', 41: 'cup', 42: 'fork', 43: 'knife', 44: 'spoon', 45: 'bowl', 46: 'banana', 47: 'apple', 48: 'sandwich', 49: 'orange', 50: 'broccoli', 51: 'carrot', 52: 'hot dog', 53: 'pizza', 54: 'donut', 55: 'cake', 56: 'chair', 57: 'couch', 58: 'potted plant', 59: 'bed', 60: 'dining table', 61: 'toilet', 62: 'tv', 63: 'laptop', 64: 'mouse', 65: 'remote', 66: 'keyboard', 67: 'cell phone', 68: 'microwave', 69: 'oven', 70: 'toaster', 71: 'sink', 72: 'refrigerator', 73: 'book', 74: 'clock', 75: 'vase', 76: 'scissors', 77: 'teddy bear', 78: 'hair drier', 79: 'toothbrush'}

# load the model from disk and grab all input image paths
model = models.load_model(args["model"], backbone_name="resnet50")

# initialize the frame dimensions (we'll set them as soon as we read
# the first frame from the video)

print("[INFO] starting video stream...")
vs = cv2.VideoCapture(args["input"])
outroot = args["output"]
fps = FPS().start()

a = datetime.datetime.utcnow()
b = a.strftime("%m%d%Y%H%M%S")
video_path = outroot + "/output_{}.avi".format(b)
out = None
frame_counter = 0
skip_frames = args['skipframes']

def Zoom(cv2Object, zoomSize):
    # Resizes the image/video frame to the specified amount of "zoomSize".
    # A zoomSize of "2", for example, will double the canvas size
    cv2Object = imutils.resize(cv2Object, width=(zoomSize * cv2Object.shape[1]))
    # center is simply half of the height & width (y/2,x/2)
    center = (int(cv2Object.shape[0]/2),int(cv2Object.shape[1]/2))
    # cropScale represents the top left corner of the cropped frame (y/x)
    cropScale = (int(center[0]/zoomSize), int(center[1]/zoomSize))
    # The image/video frame is cropped to the center with a size of the original picture
    # image[y1:y2,x1:x2] is used to iterate and grab a portion of an image
    # (y1,x1) is the top left corner and (y2,x1) is the bottom right corner of new cropped frame.
    cv2Object = cv2Object[cropScale[0]:(center[0] + cropScale[0]), cropScale[1]:(center[1] + cropScale[1])]
    return cv2Object

people_list = []
# loop over the frames from the video stream
while True:
	# grab the frame from the threaded video stream and resize it
	# to have a maximum width of 400 pixels
	res, frame = vs.read()
	if res == False:
		break

	H = frame.shape[0]
	W = frame.shape[1]

	# load the input image (in BGR order), clone it, and preprocess it
	#image = read_image_bgr(image)
	#frame = Zoom(frame,3)
	image = imutils.resize(frame, width = 640)
	output = image.copy()

	#initiate video writer
	if not out:
		out = cv2.VideoWriter(video_path,
					cv2.VideoWriter_fourcc('M','J','P','G'), 10, (image.shape[1],image.shape[0]))

	#do we want to process this frame?
	if frame_counter % skip_frames == 0:
		image = preprocess_image(image)
		(image, scale) = resize_image(image)
		image = np.expand_dims(image, axis=0)

		# detect objects in the input image and correct for the image scale
		(boxes, scores, labels) = model.predict_on_batch(image)
		boxes /= scale

		#img_undistorted = cv2.fisheye.undistortImage(image, K, D=D, Knew=Knew)

		# loop over the detections
		for (box, score, label) in zip(boxes[0], scores[0], labels[0]):
			# filter out weak detections
			#print("detection!")
			if score < args["confidence"]:
				continue

			#print('detection')
			if LABELS[label] != 'person':
				continue

			print("Person Detected...")

			# convert the bounding box coordinates from floats to integers
			box = box.astype("int")

			# crop person
			xmin = int(box[0])
			ymin = int(box[1])
			xmax = int(box[2])
			ymax = int(box[3])

			people_list.append(box)
			print(box)

			# build the label and draw the label + bounding box on the output
			# image
			#label = "{}: {:.2f}".format(label, score)
	for people in people_list:
		cv2.rectangle(output, (people[0], people[1]), (people[2], people[3]),
			(0,255,0), 3) # person
			#cv2.putText(output, label, (box[0], box[1] - 10),
			#	cv2.FONT_HERSHEY_SIMPLEX, 0.5, colour, 2) # label

	out.write(output)

	#show image
	cv2.imshow('frame',output)

	key = cv2.waitKey(1) & 0xFF
	frame_counter += 1

	# if the `q` key was pressed, break from the loop
	if frame_counter % 50 == 0:
		print("Processed {} frames".format(frame_counter))

	if key == ord("q"):
		break

	# update the FPS counter
	fps.update()

# stop the timer and display FPS information
fps.stop()
print("[INFO] elapsed time: {:.2f}".format(fps.elapsed()))
print("[INFO] approx. FPS: {:.2f}".format(fps.fps()))


# do a bit of cleanup
out.release()
cv2.destroyAllWindows()
