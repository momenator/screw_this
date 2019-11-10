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
font2 = pygame.font.SysFont("arial", 40)


def answer(val):
    if val == 0:
        print('correct!')
    noPause = True
    for i in range(0,4):
        screen.fill(WHITE)
        _, frame = cap.read()
        cv2.imwrite(filename='./images/latest_image.jpg', img=frame)
        im1 = pygame.image.load('./images/latest_image.jpg')
        screen.blit(im1, (0,0))
        clear = font2.render("Clear table for next batch in " + str(3-i) + " seconds", True, (50, 255, 50))
        screen.blit(clear, (350,500))
        pygame.display.flip()
        time.sleep(1.3)
        if i == 3:
            screen.fill(WHITE)
            captu = font2.render("Capturing...", True, (0, 0, 0))
            screen.blit(captu, (350,300))
            pygame.display.flip()


# The clock will be used to control how fast the screen updates
clock = pygame.time.Clock()

PERIOD = 5
PIXEL_WIDTH = 24.25
cap = cv2.VideoCapture(0)
last_Measurements = []


gameloop = True
noPause = True

# Take a picture every second
while gameloop == True:
    # stop for 1 second
    time.sleep(PERIOD)
        

    screen.fill(WHITE)

    presentimages = ['Load Image', 'Convert to Gray', 'GaussianBlur', 'Edge Detection', 'Dilate Edges', 'Erode Edges', 'Bounding Box Computed']

    # --- Go ahead and update the screen with what we've drawn.

    _, frame = cap.read()

    cv2.imwrite(filename='./images/latest_image.jpg', img=frame)

    #frame = cv2.imread('./images/coin.jpg')

    if noPause == True:
        # get measurements here!
        mea = get_measurements('./images/latest_image.jpg', PIXEL_WIDTH)
        if len(mea) < len(last_Measurements):
            noPause = False
        else:
            last_Measurements = mea
            for item in mea:
                string = str(item[1]) + "cm long x " + str(item[2]) + "cm wide "
                if item[0] == 0:
                    string = string + "concrete drill"
                else:
                    string = string + "metal drill"

            newtext = font.render(string, True, (0, 0, 0))
            screen.blit(newtext, (500, 300))
            x=300
            y=300
            yes_button = pygame.draw.rect(screen,(0,240,0),(150+x,160+y,100,50))
            yes = font.render("Correct", True, (255, 255, 255))
            screen.blit(yes, (150+x+10,160+y+10))
            no_button = pygame.draw.rect(screen,(240,0,0),(500+x,160+y,100,50))
            no = font.render("Incorrect", True, (255, 255, 255))
            screen.blit(no, (500+x+10,160+y+10))
            for index,name in enumerate(presentimages):
                if path.exists('images/present/'+ str(index+1) +".jpg"):
                    loadedimg = pygame.image.load('images/present/'+ str(index+1) +".jpg")
                    newimg = pygame.transform.scale(loadedimg, (160,120))
                    text = font.render(name, True, (0, 0, 0))
                    screen.blit(newimg, (0,index*120))
                    screen.blit(text, (160,index*120+40))

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

        # --- Main event loop
    for event in pygame.event.get(): # User did something
        if event.type == pygame.QUIT: # If user clicked close
                gameloop = False # Flag that we are done so we exit this loop
        elif event.type == pygame.MOUSEBUTTONDOWN:
            # 1 is the left mouse button, 2 is middle, 3 is right.
            if pygame.mouse.get_pos() >= (450,460):
                if pygame.mouse.get_pos() <= (550,510):
                    # Increment the number.
                    answer(0)
            if pygame.mouse.get_pos() >= (800,230):
                if pygame.mouse.get_pos() <= (900,280):
                    # Increment the number.
                    answer(1)
            

    pygame.display.flip()

# window.Close()