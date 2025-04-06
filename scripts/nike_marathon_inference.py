# import the necessary packages
from pyimagesearch.centroidtracker import CentroidTracker
from pyimagesearch.trackableobject import TrackableObject
from keras_retinanet.utils.image import preprocess_image
from keras_retinanet.utils.image import read_image_bgr
from keras_retinanet.utils.image import resize_image
from imutils.video import VideoStream
from keras.models import load_model
from keras_retinanet import models
from imutils.video import FPS
from datetime import datetime
import pandas as pd
import numpy as np
import scipy.misc
import argparse
import imutils
import time
import dlib
import cv2

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-pm", "--prototxt", default="resnet50_coco_best_v2.1.0.h5",
	help="path to people model") #file1
ap.add_argument("-sm", "--shoemodel", default="model_20200316_40.h5",
	help="path to shoe model") #file1
ap.add_argument("-v", "--video", type=str,
	help="path to input video file")
ap.add_argument("-o", "--output", type=str, default = None,
	help="path to optional output video file")
ap.add_argument("-c", "--confidence", type=float, default=0.5,
	help="minimum probability to filter weak detections")
ap.add_argument("-l", "--label", default='person',
	help="class label we are interested in detecting + tracking")
ap.add_argument("-s", "--skip-frames", type=int, default=30,
	help="# of skip frames between detections")
args = vars(ap.parse_args())

# initialize the list of class labels MobileNet SSD was trained to
# detect
LABELS = {0: 'person', 1: 'bicycle', 2: 'car', 3: 'motorcycle', 4: 'airplane', 5: 'bus', 6: 'train',
7: 'truck', 8: 'boat', 9: 'traffic light', 10: 'fire hydrant', 11: 'stop sign', 12: 'parking meter',
13: 'bench', 14: 'bird', 15: 'cat', 16: 'dog', 17: 'horse', 18: 'sheep', 19: 'cow', 20: 'elephant',
21: 'bear', 22: 'zebra', 23: 'giraffe', 24: 'backpack', 25: 'umbrella', 26: 'handbag', 27: 'tie',
28: 'suitcase', 29: 'frisbee', 30: 'skis', 31: 'snowboard', 32: 'sports ball', 33: 'kite', 34: 'baseball bat',
35: 'baseball glove', 36: 'skateboard', 37: 'surfboard', 38: 'tennis racket', 39: 'bottle', 40: 'wine glass',
41: 'cup', 42: 'fork', 43: 'knife', 44: 'spoon', 45: 'bowl', 46: 'banana', 47: 'apple', 48: 'sandwich',
49: 'orange', 50: 'broccoli', 51: 'carrot', 52: 'hot dog', 53: 'pizza', 54: 'donut', 55: 'cake',
56: 'chair', 57: 'couch', 58: 'potted plant', 59: 'bed', 60: 'dining table', 61: 'toilet', 62: 'tv',
63: 'laptop', 64: 'mouse', 65: 'remote', 66: 'keyboard', 67: 'cell phone', 68: 'microwave', 69: 'oven',
70: 'toaster', 71: 'sink', 72: 'refrigerator', 73: 'book', 74: 'clock', 75: 'vase', 76: 'scissors',
77: 'teddy bear', 78: 'hair drier', 79: 'toothbrush'}

SHOE_LABELS = {0:'0',1:'1',2:'2',3:'3',4:'4'}

# load our serialized model from disk
print("[INFO] loading model...")
people_model = models.load_model('C:/Users/pressma/OneDrive - EY/Documents/Jobs/nike/resnet50_coco_best_v2.1.0.h5', backbone_name="resnet50")
shoe_model = models.load_model(args["shoemodel"], backbone_name="resnet50")

# initialize the video stream and output video writer
print("[INFO] starting video...")
master_label = args['label']

#define area function to test the size of people relative to the frame
def box_compare(box,dim1,dim2,thresh=0.015):
	sX, sY, eX, eY = box.astype("int")
	b_area = abs(eY-sY)*abs(eX-sX)
	f_area = dim1*dim2
	print(b_area/f_area)
	if b_area/f_area < thresh:
		return "Stop"
	return "Go"

vids = [args["video"]]

for VID in vids:

	VID_path = VID

	#write filename based on timestamp
	now = datetime.now()
	filename = now.strftime("%Y%m%d%H%M%S")+ "_" +str(VID[:-4])
	print("Saving using filename: {}".format(filename))

	print("[INFO] opening {}...".format(filename))
	vs = cv2.VideoCapture(VID_path)

	# initialize the video writer (we'll instantiate later if need be)
	writer = None
	# initialize the frame dimensions (we'll set them as soon as we read
	# the first frame from the video)
	W = None
	H = None
	# instantiate our centroid tracker, then initialize a list to store
	# each of our dlib correlation trackers, followed by a dictionary to
	# map each unique object ID to a TrackableObject
	ct = CentroidTracker(maxDisappeared=80, maxDistance=250)
	trackers = []
	trackableObjects = {}
	# initialize the total number of frames processed thus far, along
	# with the total number of objects that have moved either up or down
	totalFrames = 0
	totalDown = 0
	totalUp = 0
	# start the frames per second throughput estimator
	fps = FPS().start()
	#initiate frame width
	frame_width = 500
	shoe_width = 500


	#initiate dataframes
	#for brand
	brand_id = int(0)
	headers1 = ['brand_id','frame','brand_detected','centx','centy','conf']
	brand_detections = pd.DataFrame(columns=headers1)
	#for tracking
	tracking_id = int(0)
	headers2 = ['tracking_id','frame','centx','centy','object_id']
	people_tracking = pd.DataFrame(columns=headers2)

	# loop over frames from the video stream
	while True:
		# grab the next frame and handle if we are reading from either
		# VideoCapture or VideoStream
		frame = vs.read()
		frame = frame[1]

		# if we are viewing a video and we did not grab a frame then we
		# have reached the end of the video
		if VID is not None and frame is None:
			break

		# if we are viewing a video and we did not grab a frame then we
		# have reached the end of the video
		if VID is not None and frame is None:
			break

		# resize the frame to have a maximum width of 500 pixels (the
		# less data we have, the faster we can process it), then convert
		# the frame from BGR to RGB for dlib
		aspect = frame.shape[1]/frame_width
		image = imutils.resize(frame, width = frame_width)
		output = image.copy()
		rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
		# if the frame dimensions are empty, set them
		if W is None or H is None:
			(H, W) = output.shape[:2]

		# if we are supposed to be writing a video to disk, initialize
		# the writer
		if writer is None and output is not None:
			fourcc = cv2.VideoWriter_fourcc(*"MJPG")
			writer = cv2.VideoWriter(filename+".avi", fourcc, 30,
				(W, H), True)

		# initialize the current status along with our list of bounding
		# box rectangles returned by either (1) our object detector or
		# (2) the correlation trackers
		status = "Waiting"
		rects = []
		process = False

		# check to see if we should run a more computationally expensive
		# object detection method to aid our tracker
		if totalFrames % args["skip_frames"] == 0:
			# set the status and initialize our new set of object trackers
			status = "Detecting"
			trackers = []
			process = True


			# convert the frame to a blob and pass the blob through the
			# network and obtain the detections
			image = preprocess_image(image)
			(image, scale) = resize_image(image)
			image = np.expand_dims(image, axis=0)

			# detect objects in the input image and correct for the image scale
			(boxes, scores, labels) = people_model.predict_on_batch(image)
			boxes /= scale

			# loop over the detections
			for (box, score, label) in zip(boxes[0], scores[0], labels[0]):
				## check confidence level
				if score < args["confidence"]:
					continue

				## only looking at people
				if LABELS[label] != master_label:
					continue

				## check if person is larger than minimum size
				go_ahead = box_compare(box,output.shape[0],output.shape[1],0.015)
				if go_ahead == "Stop":
					continue

				# compute the (x, y)-coordinates of the bounding box
				# for the object
				(startX, startY, endX, endY) = box.astype("int")
				cv2.rectangle(output, (startX, startY), (endX, endY), (0, 255, 0), 2)

				# construct a dlib rectangle object from the bounding
				# box coordinates and then start the dlib correlation
				# tracker
				tracker = dlib.correlation_tracker()
				rect = dlib.rectangle(startX, startY, endX, endY)
				tracker.start_track(rgb, rect)
				# add the tracker to our list of trackers so we can
				# utilize it during skip frames
				trackers.append(tracker)

				#now crop of legs/shoes
				xmin = int(box[0]*aspect)
				xmax = int(box[2]*aspect)
				ymax = int(box[3]*aspect)
				ymin = int(box[1]*aspect)
				heighter = ymax-ymin
				ymin = int(ymin + heighter*0.5)

				#list centroids
				centx = int((box[0]+box[2])/2)
				centy = int((box[3]+box[1])/2)

				tmpFrame = frame.copy()
				crop_img = tmpFrame[ymin:ymax,xmin:xmax]

				#resize shoe image and process for retinanet
				shoe_image = imutils.resize(crop_img, width = shoe_width)
				shoe_image = preprocess_image(shoe_image)
				(shoe_image, Sscale) = resize_image(shoe_image)
				shoe_image = np.expand_dims(shoe_image, axis=0)

				# detect objects in the input image and correct for the image scale
				(Sboxes, Sscores, Slabels) = shoe_model.predict_on_batch(shoe_image)
				Sboxes /= Sscale

				# loop over the detections
				for (Sbox, Sscore, Slabel) in zip(Sboxes[0], Sscores[0], Slabels[0]):
					if Sscore < args["confidence"]+0.2:
						continue

					brand = SHOE_LABELS[Slabel]
					#tmpFrame2 = frame.copy()

					#prepare the brand image
					brand_id += 1
					shoetect = {'brand_id':brand_id,'frame':totalFrames,'brand_detected':brand,
								'centx':centx,'centy':centy,'conf':Sscore}
					brand_detections = brand_detections.append(shoetect,ignore_index=True)

					## Print the shoe brand discovered
					print("Found: {} Confidence: {}".format(brand,Sscore))

					##now crop of logo on shoe
					#Sxmin = int(Sbox[0])
					#Sxmax = int(Sbox[2])
					#Symax = int(Sbox[3])
					#Symin = int(Sbox[1])

					#saver = tmpFrame2[(Symin+ymin):(Symax+ymin),(Sxmin+xmin):(Sxmax+xmin)]
					save_path = "shoes/{}/{}.jpg".format(brand,filename+"_"+brand+"_"+str(brand_id))

					cv2.imwrite(save_path,crop_img)

					#draw on the shoe image
					#cv2.rectangle(crop_img, (Sxmin, Symin), (Sxmax, Symax), (0, 0, 255), 2)
					#cv2.putText(output, "{}".format(brand), (Sxmin, Symin),  cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)

				#show the shoe with the brand drawn on
				cv2.imshow("shoe_image",crop_img)

		# otherwise, we should utilize our object *trackers* rather than
		# object *detectors* to obtain a higher frame processing throughput
		else:
			# loop over the trackers
			for tracker in trackers:
				# set the status of our system to be 'tracking' rather
				# than 'waiting' or 'detecting'
				status = "Tracking"
				# update the tracker and grab the updated position
				tracker.update(rgb)
				pos = tracker.get_position()
				# unpack the position object
				startX = int(pos.left())
				startY = int(pos.top())
				endX = int(pos.right())
				endY = int(pos.bottom())
				# add the bounding box coordinates to the rectangles list
				rects.append((startX, startY, endX, endY))
				cv2.rectangle(output, (startX, startY), (endX, endY), (0, 255, 0), 2)

		# draw a horizontal line in the center of the frame -- once an
		# object crosses this line we will determine whether they were
		# moving 'up' or 'down'
		prop = 2
		#cv2.line(output, (0, H // prop), (W, H // prop), (0, 255, 255), 2)
		# use the centroid tracker to associate the (1) old object
		# centroids with (2) the newly computed object centroids
		objects = ct.update(rects)
		Tcents = []

		# loop over the tracked objects
		for (objectID, centroid) in objects.items():
			# check to see if a trackable object exists for the current
			# object ID
			to = trackableObjects.get(objectID, None)
			# if there is no existing trackable object, create one
			if to is None:
				to = TrackableObject(objectID, centroid)
			# otherwise, there is a trackable object so we can utilize it
			# to determine direction
			else:
				# the difference between the y-coordinate of the *current*
				# centroid and the mean of *previous* centroids will tell
				# us in which direction the object is moving (negative for
				# 'up' and positive for 'down')
				y = [c[1] for c in to.centroids]
				direction = centroid[1] - np.mean(y)
				to.centroids.append(centroid)
				# check to see if the object has been counted or not
				if not to.counted:
					# if the direction is negative (indicating the object
					# is moving up) AND the centroid is above the center
					# line, count the object
					if direction < 0 and centroid[1] < H // prop:
						totalUp += 1
						to.counted = True
					# if the direction is positive (indicating the object
					# is moving down) AND the centroid is below the
					# center line, count the object
				elif direction > 0 and centroid[1] > H // prop:
						totalDown += 1
						to.counted = True
			# store the trackable object in our dictionary
			trackableObjects[objectID] = to

			# check to see if details should be recorded
			if process == True:
				tracking_id += 1
				track = {'tracking_id':tracking_id,'centx':centroid[0],
						'centy':centroid[1],'object_id':objectID,'frame':totalFrames}
				people_tracking = people_tracking.append(track,ignore_index=True)

			# draw both the ID of the object and the centroid of the
			# object on the output frame
			text = "ID {}".format(objectID)
			cv2.putText(output, text, (centroid[0] - 10, centroid[1] - 10),
				cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
			cv2.circle(output, (centroid[0], centroid[1]), 5, (0, 255, 0), -1)

		# construct a tuple of information we will be displaying on the
		# frame
		info = [
			("Up", totalUp),
			("Down", totalDown),
			("Status", status),
		]
		# loop over the info tuples and draw them on our frame
		for (i, (k, v)) in enumerate(info):
			text = "{}: {}".format(k, v)
			#cv2.putText(output, text, (10, H - ((i * 20) + 20)),
				#cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

		# check to see if we should write the frame to disk
		if writer is not None and output is not None:
			writer.write(output)
		# show the output frame
		cv2.imshow("Frame", output)
		key = cv2.waitKey(1) & 0xFF
		# if the `q` key was pressed, break from the loop
		if key == ord("q"):
			break
		# increment the total number of frames processed thus far and
		# then update the FPS counter
		totalFrames += 1
		fps.update()

	# stop the timer and display FPS information
	fps.stop()
	print("[INFO] elapsed time: {:.2f}".format(fps.elapsed()))
	print("[INFO] approx. FPS: {:.2f}".format(fps.fps()))
	# check to see if we need to release the video writer pointer
	if writer is not None:
		writer.release()
	# if we are not using a video file, stop the camera video stream
	#if not args.get("video", False):
	#	vs.stop()
	# otherwise, release the video file pointer
	#else:
	#	vs.release()
	# close any open windows
	cv2.destroyAllWindows()


	##### ------------------ COMMENCE POST-PROCESSING

	#create tuple list of detections
	brand_subset = brand_detections[['brand_detected', 'frame', 'centx','centy']]
	brand_tuples = [tuple(x) for x in brand_subset.to_numpy()]
	print(brand_tuples)

	# assign brands from brand frame to tracking frame
	def assign_brand(row):
		margin = 10
		default = "Other"
		frame = row['frame']
		centx = row['centx']
		centy = row['centy']
		for a,b,c,d in brand_tuples:
			if abs(frame-b) < margin:
				if abs(centx-c) < margin:
					if abs(centy-d) < margin:
						return a
		return default

	#apply to tracking DataFrame
	people_tracking['brand'] = people_tracking.apply(assign_brand, axis=1)

	#pivot to do brand lookup per person tracked
	people_brands = people_tracking[people_tracking['brand'] != 'Other']
	people_brands = people_brands[['object_id','brand']]
	pivot2 = people_brands.groupby(['object_id']).agg(lambda x:x.value_counts().index[0])

	#generalise all brands based on ID
	def gen_brand(row):
		default = 'Other'
		id = row['object_id']
		try:
			val = pivot2.loc[id][0]
			return val
		except:
			return default

	# apply gender generalisation
	people_tracking['gen_brand'] = people_tracking.apply(gen_brand, axis=1)

	# if a video path was not supplied, grab a reference to the webcam
	people_tracking['filepath'] = VID
	brand_detections['filepath'] = VID

	print(people_tracking.head(10))
	print(brand_detections.head(10))

	#write out output
	people_tracking.to_csv("{}_people_tracking.csv".format(filename))
	brand_detections.to_csv("{}_brand_detections.csv".format(filename))
