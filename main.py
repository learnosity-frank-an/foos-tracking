# USAGE
# python motion_detector.py
# python motion_detector.py --video videos/example_01.mp4

# import the necessary packages
import argparse
import datetime
import imutils
import time
import cv2
import threading
import numpy
from collections import deque

frames = 0
framesPerSec = 0

def printFps():
    global frames
    global framesPerSec
    framesPerSec = frames
    frames = 0
    threading.Timer(1.0, printFps).start()


def calculateDegree(currentHeight):
    actualLengthCM = 5
    currentLengthCM = float(actualLengthCM) * float(currentHeight) / 129
    arcSinVal = numpy.arcsin(currentLengthCM / actualLengthCM)
    return 90 * arcSinVal / float(numpy.pi / 2)

# construct the argument parser and parse the arguments
print cv2.__version__
ap = argparse.ArgumentParser()
ap.add_argument("-v", "--video", help="path to the video file")
ap.add_argument("-a", "--min-area", type=int, default=500, help="minimum area size")
args = vars(ap.parse_args())

# if the video argument is None, then we are reading from webcam
if args.get("video", None) is None:
    camera = cv2.VideoCapture(0)
    time.sleep(0.25)

# otherwise, we are reading from a video file
else:
    camera = cv2.VideoCapture(args["video"])

# initialize the first frame in the video stream
firstFrame = None
printFps()

redLower = (0, 16, 0)
redUpper = (10, 255, 255)

yellowLower = (18, 41, 0)
yellowUpper = (61, 255, 255)

pts = deque(maxlen=32)
counter = 0
(dX, dY) = (0, 0)
direction = ""
# loop over the frames of the video
while True:
    # grab the current frame and initialize the occupied/unoccupied
    # text
    (grabbed, frame) = camera.read()
    frames += 1

  #  frame = imutils.resize(frame, width=600)
    blurred = cv2.GaussianBlur(frame, (11, 11), 0)
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # construct a mask for the color "green", then perform
    # a series of dilations and erosions to remove any small
    # blobs left in the mask
    #mask = cv2.inRange(hsv, redLower, redUpper)
    mask = cv2.inRange(hsv, yellowLower, yellowUpper)
    mask = cv2.erode(mask, None, iterations=2)
    mask = cv2.dilate(mask, None, iterations=2)

    cv2.imshow("Thresh", mask)

    # find contours in the mask
    (_, cnts, _) = cv2.findContours(
        mask.copy(),
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE)
    center = None
    # only proceed if at least one contour was found
    if len(cnts) > 0:
        # find the largest contour in the mask, then use
        # it to compute the minimum enclosing circle and
        # centroid
        c = max(cnts, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(c)
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(frame, "Height: {:10.4f}".format(h), (10, 35),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
        cv2.putText(frame, "Degree: {:10.4f}".format(calculateDegree(h)), (10, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

    cv2.putText(frame, "FPS: {}".format(framesPerSec), (10, 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
# show the frame and record if the user presses a key
    cv2.imshow("Camera 1", frame)  # cleanup the camera and close any open windows

camera.release()
cv2.destroyAllWindows()
