## IMPORTS
import torch
import yaml
import window
import board 
import chess
import train 
import asyncio
import queue
import pygame
import time

## GLOBALS
with open('config.yaml', 'r') as file:
    CONFIG = yaml.safe_load(file) # CONFIG stores important system configurations

def select_action(board, fogBot):
    state = board.state.to("cpu")
    values = fogBot(state)
    mask = train.get_legal_move_mask(board)
    values = values * mask
    action = torch.argmax(values)
    move = train.convert_action_to_move(action, board)
    return move

## play class allows a user to play chess against fog-bot in either the normal or fogged variation
## TODO add fogged variation of chess 
def main():
    ## SET UP VARIABLES FOR MAIN PLAYING LOOP
    model_fogBot = torch.load("models/"+ CONFIG['model_path'],  map_location=torch.device('cpu')) ## stores model of bot we are playing against
    chess_board = board.CustomBoard("cpu") ## initializes the game board
    game_window = window.Window(True)
    user_color, fog_on = game_intro(game_window) ## grab color of player
    turn = False ## true if white, false if black
    bot_state = None
    play_itself = CONFIG['play_itself']
    ## MAIN PLAYING LOOP x
    while(not chess_board.is_game_over()):
        turn = not turn
        # print(chess_board.board)
        game_window.display_board(chess_board, bot_state, fog_on,user_color) ## display boards to user
        pygame.display.flip()

        if(turn == user_color): # get user move if it is user's turn
            if play_itself:
                move = select_action(chess_board, model_fogBot)
                chess_board.update_move(move)
                time.sleep(2)
                continue
            move = (prompt_user_move(game_window, chess_board))
            try:
                chess_board.update_move(move)
            except:
                print("INVALID MOVE! TRY AGAIN")
                turn = not turn
                continue 
            
        else:
            bot_state = chess_board.state.to("cpu")

            bot_move = select_action(chess_board, model_fogBot)
            chess_board.update_move(bot_move)
        
        if play_itself:
            time.sleep(2)
    
                   
    
    
def game_intro(game_window):
    return game_window.introduction() ## stores if user is playing as white or not

    
def prompt_user_move(game_window, chess_board):
    return game_window.prompt_user_move()
    
    



if __name__=="__main__":
    main()