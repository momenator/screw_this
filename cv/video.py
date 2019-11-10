import time
import cv2
import numpy as np
from processing import get_measurements
# import gui
import PySimpleGUI as sg
import pygame
from os import path
pygame.init()

# Define some colors
BLACK = ( 0, 0, 0)
WHITE = ( 255, 255, 255)
GREEN = ( 0, 255, 0)
RED = ( 255, 0, 0)

# Open a new window
size = (1200, 900)
screen = pygame.display.set_mode(size)
pygame.display.set_caption("Part Identifier")
font = pygame.font.SysFont("arial", 18)


# The clock will be used to control how fast the screen updates
clock = pygame.time.Clock()

PERIOD = 2
PIXEL_WIDTH = 24.25
cap = cv2.VideoCapture(0)
last_Measurements = []

gameloop = True

# Take a picture every second
while gameloop == True:
    # stop for 1 second
    time.sleep(PERIOD)

    # --- Main event loop
    for event in pygame.event.get(): # User did something
        if event.type == pygame.QUIT: # If user clicked close
                gameloop = False # Flag that we are done so we exit this loop

    screen.fill(WHITE)

    presentimages = ['Load Image', 'Convert to Gray', 'GaussianBlur', 'Edge Detection', 'Dilate Edges', 'Erode Edges', 'Bounding Box Computed']

    for index,name in enumerate(presentimages):
        if path.exists('images/present/'+ str(index+1) +".jpg"):
            loadedimg = pygame.image.load('images/present/'+ str(index+1) +".jpg")
            newimg = pygame.transform.scale(loadedimg, (160,120))
            text = font.render(name, True, (0, 0, 0))
            screen.blit(newimg, (0,index*120))
            screen.blit(text, (160,index*120+40))

    # --- Go ahead and update the screen with what we've drawn.

    _, frame = cap.read()

    cv2.imwrite(filename='./images/latest_image.jpg', img=frame)

    #frame = cv2.imread('./images/coin.jpg')

    # get measurements here!
    mea = get_measurements('./images/latest_image.jpg', PIXEL_WIDTH)
    for item in mea:
        string = str(item[1]) + "cm long x " + str(item[2]) + "cm wide "
        if item[0] == 0:
           string = string + "concrete drill"
        else:
            string = string + "metal drill"

    newtext = font.render(string, True, (0, 0, 0))
    screen.blit(newtext, (500, 300))

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    pygame.display.flip()

window.Close()