## WINDOW CLASS utilizes the pygame library to display PNG images on a window
## we utilize this to display the simulated chess games our model plays by updating
## the window with an image of the current board position everytime a move is made
## TODO- must add Fog of war/covering of unseen tiles
import pygame
import os
import time
import events 
import chess
import board


## GLOBAL VARIABLES
IMG_DIR = 'chess_pieces/'
SCREEN_WIDTH, SCREEN_HEIGHT = 1040, 512
BOARD_WIDTH, BOARD_HEIGHT = 512, 512
PIECE_WIDTH, PIECE_HEIGHT = 32,32
TILE_WIDTH = BOARD_WIDTH / 8
TILE_HEIGHT = BOARD_HEIGHT / 8
PIECE_SIZE = (PIECE_WIDTH, PIECE_HEIGHT)
FOGGED = False
TILES_PER_ROW, TOTAL_ROWS = 8,8
BOARD_ONE = 0 
BOARD_TWO = 528 
LARGE_BUTTON_HEIGHT = 128
LARGE_BUTTON_WIDTH = 256
LARGE_TEXT_SIZE = 24
SMALL_BUTTON_WIDTH = 80
SMALL_BUTTON_HEIGHT = 80
## COLORS
GREY = (128, 128, 128)
RED = (255, 0, 0)
GREEN = (0, 255, 0)
BLUE = (0, 0, 255)
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
BROWN = (165, 42, 42)
BEIGE = (245, 245, 220)
DARK_GREY = (64,64,64)
SCREEN = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT)) ## Display Screen
INTRO_STRING = "Welcome to chess vs fogbot!"
pygame.display.set_caption("fogbot chess match")

class Window():

    def __init__ (self, fog):
        pass
    
    ## draws a basic black and white chess board on a WDITH/HEIGHT plane
    def draw_base_board(self, BOARD):

        
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
            tile = pygame.Rect((tile_row * TILE_WIDTH) + BOARD, tile_column * TILE_HEIGHT, TILE_WIDTH, TILE_HEIGHT)
            pygame.draw.rect(SCREEN, tile_color, tile)
            #print(f"Placed tile at ({tile_row},{tile_column})")
            
    def draw_pieces_board_one(self, FEN_STRING):
        self.draw_pieces(FEN_STRING, BOARD_ONE)
        
    def draw_base_board_one(self):
        self.draw_base_board(BOARD_ONE)
        
    def draw_fog_board_one(self, chess_board):
        self.draw_fog(chess_board, BOARD_ONE)
    
    ## SEAN MODIFY THESE ONES
    def draw_pieces_board_two(self, FEN_STRING):
        self.draw_pieces(FEN_STRING, BOARD_TWO)
    
    def draw_base_board_two(self):
        self.draw_base_board(BOARD_TWO)
        
    def draw_fog_board_two(self, chess_board):
        self.draw_fog(chess_board, BOARD_TWO)
    
        
    
    def draw_pieces(self, FEN_string, BOARD):
        current_board_pos = 0
        for char in FEN_string:
            curr_row = current_board_pos % 8
            curr_coloumn = current_board_pos // 8
            if (char == " "):
                return        
            elif (char == 'R'):
                #print(f"placing a white rook at ({curr_row}, {curr_coloumn})")
                self.draw_piece(IMG_DIR + 'white_rook.png', curr_row, curr_coloumn, BOARD)
                current_board_pos += 1
            elif (char == 'r'): 
                #print(f"placing a black rook at ({curr_row}, {curr_coloumn})")
                self.draw_piece(IMG_DIR + 'black_rook.png', curr_row, curr_coloumn, BOARD)
                current_board_pos += 1
            elif (char == 'P'):
                #print(f"placing a white pawn at ({curr_row}, {curr_coloumn})")
                self.draw_piece(IMG_DIR + 'white_pawn.png', curr_row, curr_coloumn, BOARD)
                current_board_pos += 1
            elif (char == 'p'):
                #print(f"placing a black pawn at ({curr_row}, {curr_coloumn})")
                self.draw_piece(IMG_DIR + 'black_pawn.png', curr_row, curr_coloumn, BOARD)
                current_board_pos += 1
            elif (char == 'N'):
                #print(f"placing a white knight at ({curr_row}, {curr_coloumn})")
                self.draw_piece(IMG_DIR + 'white_knight.png', curr_row, curr_coloumn, BOARD)
                current_board_pos += 1
            elif (char == 'n'):
                #print(f"placing a black knight at ({curr_row}, {curr_coloumn})")
                self.draw_piece(IMG_DIR + 'black_knight.png', curr_row, curr_coloumn, BOARD)
                current_board_pos += 1
            elif (char == 'B'):
                #print(f"placing a white bishop at ({curr_row}, {curr_coloumn})")
                self.draw_piece(IMG_DIR + 'white_bishop.png', curr_row, curr_coloumn, BOARD)
                current_board_pos += 1
            elif (char == 'b'):
                #print(f"placing a black bishop at ({curr_row}, {curr_coloumn})")
                self.draw_piece(IMG_DIR + 'black_bishop.png', curr_row, curr_coloumn, BOARD)
                current_board_pos += 1
            elif (char == 'Q'):
                #print(f"placing a white queen at ({curr_row}, {curr_coloumn})")
                self.draw_piece(IMG_DIR + 'white_queen.png', curr_row, curr_coloumn, BOARD)
                current_board_pos += 1
            elif (char == 'q'):
                #print(f"placing a black queen at ({curr_row}, {curr_coloumn})")
                self.draw_piece(IMG_DIR + 'black_queen.png', curr_row, curr_coloumn, BOARD)
                current_board_pos += 1
            elif (char == 'K'):
                #print(f"placing a white king at ({curr_row}, {curr_coloumn})")
                self.draw_piece(IMG_DIR + 'white_king.png', curr_row, curr_coloumn, BOARD)
                current_board_pos += 1
            elif (char == 'k'):
                #print(f"placing a black king at ({curr_row}, {curr_coloumn})")
                self.draw_piece(IMG_DIR + 'black_king.png', curr_row, curr_coloumn,BOARD)
                current_board_pos += 1
            elif(char == '/'):
                continue
            else:
                #print(f"jumping by {char} positions")
                current_board_pos += int(char)
    
    def draw_piece(self, path, row, coloumn, BOARD):
        ## load in png 
        piece_png = pygame.image.load(path)
        ## resize
        piece = pygame.transform.scale(piece_png, PIECE_SIZE)
        ## calculate location to place
        piece_location = ((row * TILE_WIDTH + (PIECE_WIDTH / 2)) + BOARD, coloumn * TILE_HEIGHT + (PIECE_HEIGHT / 2))
        ## place piece
        SCREEN.blit(piece, piece_location)
        pygame.display.flip()
        
    ## TODO - FIGURE OUT HOW TO GET LOGIC FOR IF PLAYER IS BLACK 
    def draw_fog(self, chess_board, BOARD):
        for tile_num in range(64):
            tile_row = tile_num % TILES_PER_ROW
            tile_column = tile_num // TOTAL_ROWS
            if (chess_board.is_square_fogged(chess.square(tile_row, 7 - tile_column))): ## convert from bottom left to top left 
                 fog = pygame.Rect(BOARD + (tile_row * TILE_WIDTH), tile_column * TILE_HEIGHT, TILE_WIDTH, TILE_HEIGHT)
                 pygame.draw.rect(SCREEN, GREY, fog)
    
    ## generates a board, starts a background thread and displays the board to user given that thread 
    def display_board(self, chess_board):
        self.wipe_board()
        self.draw_base_board_one()
        self.draw_pieces_board_one(chess_board.board_to_string())
        self.draw_fog_board_one(chess_board)
        self.draw_base_board_two()
        self.draw_pieces_board_two(chess_board.board_to_string())

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
        SCREEN.fill(BEIGE)
        pygame.init() ## initialize everything
        welcome_button = Custom_Button(LARGE_BUTTON_WIDTH, LARGE_BUTTON_HEIGHT, BROWN, (SCREEN_WIDTH // 2), (SCREEN_HEIGHT // 2), INTRO_STRING, WHITE, LARGE_TEXT_SIZE) 
        welcome_button.draw()
        white_button = Custom_Button(SMALL_BUTTON_WIDTH, SMALL_BUTTON_HEIGHT, DARK_GREY, (SCREEN_WIDTH * 0.75), (SCREEN_HEIGHT * 0.75), "play white", WHITE, LARGE_TEXT_SIZE)
        white_button.draw()
        black_button = Custom_Button(SMALL_BUTTON_WIDTH, SMALL_BUTTON_HEIGHT, DARK_GREY, (SCREEN_WIDTH * 0.25), (SCREEN_HEIGHT * 0.75), "play black", WHITE, LARGE_TEXT_SIZE)
        black_button.draw()
        while True:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    quit()
                elif event.type == pygame.MOUSEBUTTONDOWN:
                   if welcome_button.is_clicked(event.pos):
                       print("Button Clicked!")
                   if white_button.is_clicked(event.pos):
                        return True
                   if black_button.is_clicked(event.pos):
                       return False
        
            pygame.display.flip()
            
    ## TODO - convert from text based movement to click and drag  
    def prompt_user_move(self):
        print("please provide your move in chess notation")
        return input("what move would you like to make? ")
    
    ## TODO - create method that flips board from perspective of white to perspective of black if player chose to play as black 
    def flip_board(color):
        pass
      
    ## TODO - make graphical instead of text based
    def error_invalid_move(self):
        print("ERROR you can not make that move, it is either illegal or not in chess notation!")
    
    
    def close_game(self):
        pygame.quit()
    
class Custom_Button(Window):
    import pygame
       
    def __init__ (self, width, height, button_color, x_cord, y_cord, text, text_color, text_size):
        self.width = width
        self.height = height
        self.button_color = button_color
        self.x_cord = x_cord
        self.y_cord = y_cord
        self.text = text
        self.text_color = text_color
        self.text_size = text_size 
        print(self.x_cord)
        print(self.y_cord)
        print(self.width)
        print(self.height)
        self.rect = pygame.Rect((self.x_cord,self.y_cord), (self.width, self.height))
        self.rect.center = (self.x_cord, self.y_cord)
        print(self.rect.x)
        print(self.rect.y)
        self.button_color = button_color  
        self.font = pygame.font.Font(None, text_size)
        print(f"image position: ({self.rect.x}, {self.rect.y})")
        print(f"image size: ({self.width}, {self.height})")


    def draw(self):
        print(f"image position: ({self.rect.x}, {self.rect.y})")
        pygame.draw.rect(SCREEN, self.button_color, self.rect)
        text_surface = self.font.render(self.text, True, self.text_color)
        text_rect = text_surface.get_rect()
        text_rect.center = self.rect.center
        SCREEN.blit(text_surface, text_rect)
        
    def is_clicked(self, pos):
        return self.rect.collidepoint(pos)
            
            
            
        
        
    
        






