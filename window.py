import pygame
import pygame_widgets
from pygame_widgets.button import Button
import os
import time 
class window():



    ## GLOBAL VARIABLES

    width, height = 1000, 1000
    ## COLORS
    grey = (128, 128, 128)
    red = (255, 0, 0)
    green = (0, 255, 0)
    blue = (0, 0, 255)
    board = None ## image path to current board
    fog_on = False ## tracks if fog is on or off
    screen = None ## Display Screen

    def __init__(self, board_path=None):
        pygame.init() ## initializes import libraries
            ## construct basic board 
        background_color = (grey)
        screen = pygame.display.set_mode((width, height)) ## pixel dimensions
        pygame.display.set_caption('Fog of War Chess Board')
        screen.fill(background_color) ## fills screen with background color
        board = board_path
    

    ## GIVEN: path to png file containing new board to update too
    ## OUTCOME: Screen is modified to display new board  
    def update_board(board_path):
        time.sleep(5) ## sleep so board doesn't update constantly and we can see what is going on
        if not os.path.exists(board_path):
            print(f"Error: {board_path} does not exist")
            pygame.quit()
            quit()
        board = pygame.image.load(board_path)
        ## clear screen        
        screen.fill(grey)
        ## fill screen with gameboard
        screen.blit(board, (0, 0))
        pygame.display.flip()

    def close_game():
        pygame.quit()
        






