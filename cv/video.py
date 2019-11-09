import time
import cv2
import numpy as np
from processing import get_measurements

PERIOD = 1
PIXEL_WIDTH = 800
cap = cv2.VideoCapture(0)

# Take a picture every second
while True:

    # stop for 1 second
    time.sleep(PERIOD)

    _, frame = cap.read()
    cv2.imshow('frame', frame)

    cv2.imwrite(filename='./images/latest_image.jpg', img=frame)

    # get measurements here!
    mea = get_measurements(frame, PIXEL_WIDTH)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break