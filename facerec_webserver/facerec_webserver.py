#!/usr/bin/env python3

# import the necessary packages
from imutils.video import VideoStream
from flask import Response
from flask import Flask
from flask import render_template
from multiprocessing import Process, Manager, Lock
import argparse
# import datetime # TODO -- use to make timings of execution?

import imutils
import time
import cv2
import numpy as np
import face_recognition
import os

# initialize the output frame and a lock used to ensure thread-safe
# exchanges of the output frames (useful when multiple browsers/tabs
# are viewing the stream)
outputFrame = None
lock = Lock()
# initialize a flask object
app = Flask(__name__)

known_face_encodings = []
known_face_names = []

@app.route("/")
def index():
	# return the rendered template
	return render_template("index.html")

def load_faces():
	global known_face_encodings, known_face_names

	files = os.listdir('./faces/')
	for file in files:
		image = face_recognition.load_image_file('faces/'+file)
		known_face_encodings.append(face_recognition.face_encodings(image)[0])
		known_face_names.append(file.split('.')[0])

def recognize(outputFrame):
	# grab global references to the video stream, output frame, and
	# lock variables
	vs = cv2.VideoCapture(0)
	vs.set(cv2.CAP_PROP_FRAME_WIDTH, 320)
	vs.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)
	vs.set(cv2.CAP_PROP_FPS, 20)
	vs.set(cv2.CAP_PROP_FOURCC,cv2.VideoWriter_fourcc('M','J','P','G'))
	time.sleep(2.0)

	# loop over frames from the video stream
	while True:
		# read the next frame from the video stream, resize it,
		# convert the frame to grayscale, and blur it
		ret, frame = vs.read()
		frame = imutils.resize(frame, width=200) # DOWNSIZE!!!

		rgb_frame = frame[:, :, ::-1]

		face_locations = face_recognition.face_locations(rgb_frame)
		face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

		# Loop through each face in this frame of video
		for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
			# See if the face is a match for the known face(s)
			matches = face_recognition.compare_faces(known_face_encodings, face_encoding)

			name = "Unknown"

			# If a match was found in known_face_encodings, just use the first one.
			# if True in matches:
			#     first_match_index = matches.index(True)
			#     name = known_face_names[first_match_index]

			# Or instead, use the known face with the smallest distance to the new face
			face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
			best_match_index = np.argmin(face_distances)
			if matches[best_match_index]:
				name = known_face_names[best_match_index]

			# Draw a box around the face
			cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)

			# Draw a label with a name below the face
			cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED)
			font = cv2.FONT_HERSHEY_DUPLEX
			cv2.putText(frame, name, (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)

		with lock:
			outputFrame.append(frame.copy())

def generate():
	# grab global references to the output frame and lock variables
	global Global
	# loop over frames from the output stream
	while True:
		# wait until the lock is acquired
		with lock:
			# check if the output frame is available, otherwise skip
			# the iteration of the loop
			if len(outputFrame) == 0:
				continue
			# encode the frame in JPEG format
			(flag, encodedImage) = cv2.imencode(".jpg", outputFrame.pop())
			# ensure the frame was successfully encoded
			if not flag:
				continue
		# yield the output frame in the byte format
		yield(b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + 
			bytearray(encodedImage) + b'\r\n')

@app.route("/video_feed")
def video_feed():
	# return the response generated along with the specific media
	# type (mime type)
	return Response(generate(),
		mimetype = "multipart/x-mixed-replace; boundary=frame")

def main():
	global outputFrame, lock, app

	# construct the argument parser and parse command line arguments
	ap = argparse.ArgumentParser()
	ap.add_argument("-i", "--ip", type=str, required=True,
		help="ip address of the device")
	ap.add_argument("-o", "--port", type=int, required=True,
		help="ephemeral port number of the server (1024 to 65535)")
	args = vars(ap.parse_args())
	# start a thread that will perform motion detection
	outputFrame = Manager().list()

	load_faces()
	print('added', len(known_face_encodings), 'faces')

	p = Process(target=recognize, args=(outputFrame,))
	p.start()
	# start the flask app
	app.run(host=args["ip"], port=args["port"], debug=True,
		threaded=True, use_reloader=False)

# check to see if this is the main thread of execution
if __name__ == '__main__':
	main()
