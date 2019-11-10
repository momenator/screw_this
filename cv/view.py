import pygame
from os import path
pygame.init()

# Define some colors
BLACK = ( 0, 0, 0)
WHITE = ( 255, 255, 255)
GREEN = ( 0, 255, 0)
RED = ( 255, 0, 0)

# Open a new window
size = (700, 500)
screen = pygame.display.set_mode(size)
pygame.display.set_caption("Part Identifier")
font = pygame.font.SysFont("arial", 22)


# The loop will carry on until the user exit the game (e.g. clicks the close button).
carryOn = True

# The clock will be used to control how fast the screen updates
clock = pygame.time.Clock()

# -------- Main Program Loop -----------
while carryOn:
    # --- Main event loop
    for event in pygame.event.get(): # User did something
        if event.type == pygame.QUIT: # If user clicked close
                carryOn = False # Flag that we are done so we exit this loop

    # --- Game logic should go here

    # --- Drawing code should go here
    # First, clear the screen to white. 
    screen.fill(WHITE)

    presentimages = ['Load Image', 'Convert to Gray', 'GaussianBlur', 'Edge Detection', 'Dilate Edges', 'Erode Edges', 'Bounding Box Computed']

    for index,name in enumerate(presentimages):
        if path.exists('images/present/'+ str(index+1) +".jpg"):
            loadedimg = pygame.image.load('images/present/'+ str(index+1) +".jpg")
            newimg = pygame.transform.scale(loadedimg, (120,80))
            text = font.render(name, True, (100, 100, 0))
            screen.blit(newimg, (100,index*50-10))
            screen.blit(text, (10,index*50))

    # --- Go ahead and update the screen with what we've drawn.
    pygame.display.flip()

    # --- Limit to 60 frames per second
    clock.tick(60)

#Once we have exited the main program loop we can stop the game engine:
pygame.quit()