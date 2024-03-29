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
MEDIUM_TEXT_SIZE = 16
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
    def draw_pieces_board_two(self,bot_state, user_color):
        if bot_state is None:
            return
        
        one_hot_to_piece = {
            0: None,
            1: 'pawn',
            2: 'knight',
            3: 'bishop',
            4: 'rook',
            5: 'queen',
            6: 'king'
        }

        color_to_string = { 
            chess.WHITE: 'white',
            chess.BLACK: 'black'
        }

        self.draw_base_board(BOARD_TWO)
        bot_color = not user_color
        state = bot_state
        one_hot = state[:576]
        extra_info = state[576:]
        turn_count = extra_info[0].item()
        current_turn = extra_info[1].item()
        print(f"Turn count: {turn_count}, Current turn: {color_to_string[current_turn]}")
        one_hot = one_hot.reshape(8,8,9)

        
        for i in range(8):
            for j in range(8):
                square = chess.square(i,j)
                one_hot_at_square = one_hot[i][7-j]
                peice_info = one_hot_at_square[:7]
                fog = one_hot_at_square[8]
                if fog:
                    #draw fog
                    fog = pygame.Rect(BOARD_TWO + (i * TILE_WIDTH), j * TILE_HEIGHT, TILE_WIDTH, TILE_HEIGHT)
                    pygame.draw.rect(SCREEN, GREY, fog)
                    # check that all other encodings are 0
                    if peice_info.sum() != 0:
                        print("Fogged square has pieces on it")
                else:
                    piece = peice_info.argmax().item()
                    is_owned = one_hot_at_square[7]
                    if piece:
                        peice_type = one_hot_to_piece[piece]
                        if peice_type:
                            if is_owned:
                                color = color_to_string[bot_color]
                            else:
                                color = color_to_string[user_color]
                            piece = IMG_DIR + color + "_" + peice_type + ".png"
                            self.draw_piece(piece, i, j, BOARD_TWO)
    
    def draw_base_board_two(self):
        self.draw_base_board(BOARD_TWO)
        
    
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
    def display_board(self, chess_board, bot_state, fog_on,user_color):
        self.wipe_board()
        self.draw_base_board_one()
        self.draw_pieces_board_one(chess_board.board_to_string())
        if(fog_on):
            self.draw_fog_board_one(chess_board)
        self.draw_base_board_two()
        self.draw_pieces_board_two(bot_state, user_color)

    ## destroys background thread and clears screen before construction of new board       
    def wipe_board(self):
        ## wipe board
        SCREEN.fill(WHITE)
        
        
    ## displays five buttons prompting 
    ## user to decide whether they want to play as white or black,
    ## fog on or off, and introduces user to game 
    def introduction(self):
        SCREEN.fill(BEIGE)
        pygame.init() ## initialize everything
        welcome_button = Custom_Button(LARGE_BUTTON_WIDTH, LARGE_BUTTON_HEIGHT, BROWN, (SCREEN_WIDTH // 2), (SCREEN_HEIGHT // 2), INTRO_STRING, WHITE, LARGE_TEXT_SIZE) 
        welcome_button.draw()
        white_button = Custom_Button(SMALL_BUTTON_WIDTH, SMALL_BUTTON_HEIGHT, DARK_GREY, (SCREEN_WIDTH * 0.75), (SCREEN_HEIGHT * 0.75), "play white", WHITE, LARGE_TEXT_SIZE)
        white_button.draw()
        black_button = Custom_Button(SMALL_BUTTON_WIDTH, SMALL_BUTTON_HEIGHT, DARK_GREY, (SCREEN_WIDTH * 0.25), (SCREEN_HEIGHT * 0.75), "play black", WHITE, LARGE_TEXT_SIZE)        
        black_button.draw()
        fog_on_button = Custom_Button(SMALL_BUTTON_WIDTH, SMALL_BUTTON_HEIGHT, DARK_GREY, (SCREEN_WIDTH * 0.25), (SCREEN_HEIGHT * 0.25), "toggle fog on", WHITE, MEDIUM_TEXT_SIZE)
        fog_on_button.draw()
        fog_off_button = Custom_Button(SMALL_BUTTON_WIDTH, SMALL_BUTTON_HEIGHT, DARK_GREY, (SCREEN_WIDTH * 0.75), (SCREEN_HEIGHT * 0.25), "toggle fog off", WHITE, MEDIUM_TEXT_SIZE)
        fog_off_button.draw()
        fog_on = False
        while True:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    quit()
                elif event.type == pygame.MOUSEBUTTONDOWN:
                    if welcome_button.is_clicked(event.pos):
                        print("Button Clicked!")
                    elif white_button.is_clicked(event.pos):
                        return (True, fog_on)
                    elif black_button.is_clicked(event.pos):
                        return (False, fog_on)
                    elif fog_on_button.is_clicked(event.pos):
                        fog_on = True
                    elif fog_off_button.is_clicked(event.pos):
                        fog_on = False
        
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
            
            
            
        
        
    
        






