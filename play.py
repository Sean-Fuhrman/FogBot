## IMPORTS
import torch
import yaml
import window
import board 
import chess

## GLOBALS
with open('config.yaml', 'r') as file:
    config = yaml.safe_load(file) # CONFIG stores important system configurations



## play class allows a user to play chess against fog-bot in either the normal or fogged variation
## TODO add fogged variation of chess 
def main():
    ## model_fogBot = torch.load(config['model_path']) ## stores model of bot we are playing against
    chess_board = board.CustomBoard("cpu") ## initializes the game board
    game_window = window.Window()
    user_color = grab_color(game_window, chess_board) ## grab color of player
    while(not chess_board.is_game_over()): ## while game is ongoing grab user and computer moves and display them to user 
        game_window.update_board(chess_board.board.board_fen())
        if(chess_board.board.Turn == user_color):
            prompt_user_move(chess_board.get_possible_moves)
        else: ## TODO add way for computer to play 
            pass
                   
    
    
def grab_color(game_window, chess_board):
    game_window.user_introduction() ## welcome user 
    is_user_white = game_window.get_user_color() ## stores if user is playing as white or not
    if(is_user_white):
        player_color = chess.WHITE
    else:
        player_color = chess.BLACK
    print(player_color)



if __name__=="__main__":
    main()