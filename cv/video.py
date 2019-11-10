import time
import cv2
import numpy as np
from processing import get_measurements
# import gui
import PySimpleGUI as sg


PERIOD = 1
PIXEL_WIDTH = 24.25
cap = cv2.VideoCapture(1)

# Take a picture every second
while True:

    # stop for 1 second
    time.sleep(PERIOD)

    _, frame = cap.read()

    cv2.imwrite(filename='./images/latest_image.jpg', img=frame)

    #frame = cv2.imread('./images/coin.jpg')

    # get measurements here!
    mea = get_measurements('./images/coin.jpg', PIXEL_WIDTH)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

window.Close()