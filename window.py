## WINDOW CLASS utilizes the pygame library to display PNG images on a window
## we utilize this to display the simulated chess games our model plays by updating
## the window with an image of the current board position everytime a move is made
## TODO- must add Fog of war/covering of unseen tiles
import pygame
import pygame_widgets
from pygame_widgets.button import Button
import os
import time
import events 
import chess
import board



## GLOBAL VARIABLES
IMG_DIR = 'chess_pieces/'
WIDTH, HEIGHT = 512, 512
PIECE_WIDTH, PIECE_HEIGHT = 32,32
TILE_WIDTH = WIDTH / 8
TILE_HEIGHT = HEIGHT / 8
PIECE_SIZE = (PIECE_WIDTH, PIECE_HEIGHT)
FOGGED = False
TILES_PER_ROW, TOTAL_ROWS = 8,8
## COLORS
GREY = (128, 128, 128)
RED = (255, 0, 0)
GREEN = (0, 255, 0)
BLUE = (0, 0, 255)
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
BROWN = (165, 42, 42)
BEIGE = (245, 245, 220)
SCREEN = pygame.display.set_mode((WIDTH, HEIGHT)) ## Display Screen
pygame.display.set_caption("fogbot chess match ")

class Window():

    def __init__ (self, fog):
        pass
    
    ## draws a basic black and white chess board on a 1024/1024 plane
    def draw_base_board(self):
        SCREEN.fill(GREEN) ## clear the screen 

        
        for tile_num in range (TILES_PER_ROW * TOTAL_ROWS):
            tile_row = tile_num % TILES_PER_ROW
            tile_column = tile_num // TOTAL_ROWS 
            if(tile_row % 2 == 0):
                if(tile_column % 2 == 0):
                    tile_color = BEIGE
                else:
                    tile_color = BROWN
            else:
                if(tile_column % 2 == 0):
                    tile_color = BROWN
                else:
                    tile_color = BEIGE
            tile = pygame.Rect(tile_row * TILE_WIDTH, tile_column * TILE_HEIGHT, TILE_WIDTH, TILE_HEIGHT)
            pygame.draw.rect(SCREEN, tile_color, tile)
            #print(f"Placed tile at ({tile_row},{tile_column})")
    
    def draw_pieces(self, FEN_string):
        current_board_pos = 0
        for char in FEN_string:
            curr_row = current_board_pos % 8
            curr_coloumn = current_board_pos // 8
            if (char == " "):
                return        
            elif (char == 'R'):
                #print(f"placing a white rook at ({curr_row}, {curr_coloumn})")
                self.draw_piece(IMG_DIR + 'white_rook.png', curr_row, curr_coloumn)
                current_board_pos += 1
            elif (char == 'r'): 
                #print(f"placing a black rook at ({curr_row}, {curr_coloumn})")
                self.draw_piece(IMG_DIR + 'black_rook.png', curr_row, curr_coloumn)
                current_board_pos += 1
            elif (char == 'P'):
                #print(f"placing a white pawn at ({curr_row}, {curr_coloumn})")
                self.draw_piece(IMG_DIR + 'white_pawn.png', curr_row, curr_coloumn)
                current_board_pos += 1
            elif (char == 'p'):
                #print(f"placing a black pawn at ({curr_row}, {curr_coloumn})")
                self.draw_piece(IMG_DIR + 'black_pawn.png', curr_row, curr_coloumn)
                current_board_pos += 1
            elif (char == 'N'):
                #print(f"placing a white knight at ({curr_row}, {curr_coloumn})")
                self.draw_piece(IMG_DIR + 'white_knight.png', curr_row, curr_coloumn)
                current_board_pos += 1
            elif (char == 'n'):
                #print(f"placing a black knight at ({curr_row}, {curr_coloumn})")
                self.draw_piece(IMG_DIR + 'black_knight.png', curr_row, curr_coloumn)
                current_board_pos += 1
            elif (char == 'B'):
                #print(f"placing a white bishop at ({curr_row}, {curr_coloumn})")
                self.draw_piece(IMG_DIR + 'white_bishop.png', curr_row, curr_coloumn)
                current_board_pos += 1
            elif (char == 'b'):
                #print(f"placing a black bishop at ({curr_row}, {curr_coloumn})")
                self.draw_piece(IMG_DIR + 'black_bishop.png', curr_row, curr_coloumn)
                current_board_pos += 1
            elif (char == 'Q'):
                #print(f"placing a white queen at ({curr_row}, {curr_coloumn})")
                self.draw_piece(IMG_DIR + 'white_queen.png', curr_row, curr_coloumn)
                current_board_pos += 1
            elif (char == 'q'):
                #print(f"placing a black queen at ({curr_row}, {curr_coloumn})")
                self.draw_piece(IMG_DIR + 'black_queen.png', curr_row, curr_coloumn)
                current_board_pos += 1
            elif (char == 'K'):
                #print(f"placing a white king at ({curr_row}, {curr_coloumn})")
                self.draw_piece(IMG_DIR + 'white_king.png', curr_row, curr_coloumn)
                current_board_pos += 1
            elif (char == 'k'):
                #print(f"placing a black king at ({curr_row}, {curr_coloumn})")
                self.draw_piece(IMG_DIR + 'black_king.png', curr_row, curr_coloumn)
                current_board_pos += 1
            elif(char == '/'):
                continue
            else:
                #print(f"jumping by {char} positions")
                current_board_pos += int(char)
    
    def draw_piece(self, path, row, coloumn):
        ## load in png 
        piece_png = pygame.image.load(path)
        ## resize
        piece = pygame.transform.scale(piece_png, PIECE_SIZE)
        ## calculate location to place
        piece_location = (row * TILE_WIDTH + (PIECE_WIDTH / 2), coloumn * TILE_HEIGHT + (PIECE_HEIGHT / 2))
        ## place piece
        SCREEN.blit(piece, piece_location)
        pygame.display.flip()
        
    def draw_fog(self, chess_board):
        for tile_num in range(64):
            tile_row = tile_num % TILES_PER_ROW
            tile_column = tile_num // TOTAL_ROWS
            if (chess_board.is_square_fogged(chess.square(tile_row, tile_column))):
                 fog = pygame.Rect(tile_row * TILE_WIDTH, tile_column * TILE_HEIGHT, TILE_WIDTH, TILE_HEIGHT)
                 pygame.draw.rect(SCREEN, GREY, fog)
    
    ## generates a board, starts a background thread and displays the board to user given that thread 
    def display_board(self, FEN_NOTATION, chess_board):
        self.wipe_board()
        self.draw_base_board()
        self.draw_pieces(FEN_NOTATION)
        if(FOGGED):
            self.draw_fog(chess_board)

    ## destroys background thread and clears screen before construction of new board       
    def wipe_board(self):
        ## wipe board
        SCREEN.fill(WHITE)
        
    ## welcomes user to game
    ## TODO - make graphical instead of text based
    def user_introduction(self):
        print("welcome to chess VS. fogBot!")
        print("fogbot is a neural network taught by reinforcement learning to play fog of war and original chess")

        
    ## displays two buttons prompting 
    ## user to decide whether they want to play as white or black
    def get_user_color(self):
        while(True):
            user_choice = input("what color would you like to play as? (input w for white, b for black) ")
            if(user_choice == 'w'):
                return True
            if(user_choice == 'b'):
                return False
    ## TODO - convert from text based movement to click and drag  
    def prompt_user_move(self):
        print("please provide your move in chess notation")
        return input("what move would you like to make? ")
      
    ## TODO - make graphical instead of text based
    def error_invalid_move(self):
        print("ERROR you can not make that move, it is either illegal or not in chess notation!")
    
    
    def close_game(self):
        pygame.quit()
        
    
        






