## IMPORTS
import torch
import yaml
import window
import board 
import chess


## GLOBALS
with open('config.yaml', 'r') as file:
    CONFIG = yaml.safe_load(file) # CONFIG stores important system configurations



## play class allows a user to play chess against fog-bot in either the normal or fogged variation
## TODO add fogged variation of chess 
def main():
    ## SET UP VARIABLES FOR MAIN PLAYING LOOP
    model_fogBot = torch.load(CONFIG['model_path']) ## stores model of bot we are playing against
    chess_board = board.CustomBoard() ## initializes the game board
    game_window = window.Window()
    user_color = grab_color(game_window, chess_board) ## grab color of player
    
    ## MAIN PLAYING LOOP 
    while(not chess_board.is_game_over()):
        game_window.update_board(chess_board.board_to_string()) ## display board to user
        
        if(chess_board.get_turn() == user_color): # get user move if it is user's turn 
            move = (prompt_user_move(game_window, chess_board))
            chess_board.update_move(move)
            
        else: ## get move from bot if bot's turn 
            pass
                   
    
    
def grab_color(game_window, chess_board):
    game_window.user_introduction() ## welcome user 
    is_user_white = game_window.get_user_color() ## stores if user is playing as white or not
    if(is_user_white):
        player_color = chess.WHITE
    else:
        player_color = chess.BLACK
    print(player_color)
    
def prompt_user_move(game_window, chess_board):
    while True:
        chess_move = chess_board.board.Move.from_uci(game_window.prompt_user_move())
        if(chess_move in (chess_board.get_possible_moves())):
            return chess_move
        else:
            game_window.error_invalid_move()
    
    
    



if __name__=="__main__":
    main()