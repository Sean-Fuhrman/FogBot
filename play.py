## IMPORTS
import torch
import yaml
import window
import board 
import chess
import train 
## GLOBALS
with open('config.yaml', 'r') as file:
    CONFIG = yaml.safe_load(file) # CONFIG stores important system configurations

def select_action(board, fogBot):
    state = board.state.to("cpu")
    values = fogBot(state)
    mask = train.get_legal_move_mask(board)
    values = values * mask
    action = torch.argmax(values)
    move = train.convert_action_to_move(action)
    return move

## play class allows a user to play chess against fog-bot in either the normal or fogged variation
## TODO add fogged variation of chess 
def main():
    ## SET UP VARIABLES FOR MAIN PLAYING LOOP
    model_fogBot = torch.load("models/"+ CONFIG['model_path'],  map_location=torch.device('cpu')) ## stores model of bot we are playing against
    chess_board = board.CustomBoard("cpu") ## initializes the game board
    game_window = window.Window(True)
    user_color = grab_color(game_window, chess_board) ## grab color of player
    turn = False ## true if white, false if black
    
    ## MAIN PLAYING LOOP 
    while(not chess_board.is_game_over()):
        turn = not turn        
        game_window.display_board(chess_board.board_to_string(), chess_board) ## display board to user

        
        if(turn): # get user move if it is user's turn
            move = (prompt_user_move(game_window, chess_board))
            try:
                chess_board.update_move(move)
            except:
                print("INVALID MOVE! TRY AGAIN")
                turn = not turn
                continue 
            
        else:
            bot_move = select_action(chess_board, model_fogBot)
            chess_board.update_move(bot_move)
    
    ## display final board        
    game_window.display_board(chess_board.board_to_string())
                   
    
    
def grab_color(game_window, chess_board):
    game_window.user_introduction() ## welcome user 
    is_user_white = game_window.get_user_color() ## stores if user is playing as white or not
    if(is_user_white):
        player_color = chess.WHITE
    else:
        player_color = chess.BLACK
    print(player_color)
    
def prompt_user_move(game_window, chess_board):
    return game_window.prompt_user_move()
    
    



if __name__=="__main__":
    main()