import numpy as np

import torch as torch 
import torch.nn as nn
import torch.nn.functional as F
import board
import model

import math
import random
import matplotlib
import matplotlib.pyplot as plt
from collections import namedtuple, deque
from itertools import count

import yaml

with open('config.yaml', 'r') as file:
    config = yaml.safe_load(file)

# BATCH_SIZE is the number of transitions sampled from the replay buffer
# GAMMA is the discount factor as mentioned in the previous section
# EPS_START is the starting value of epsilon
# EPS_END is the final value of epsilon
# EPS_DECAY controls the rate of exponential decay of epsilon, higher means a slower decay
# TAU is the update rate of the target network
# LR is the learning rate of the ``AdamW`` optimizer
BATCH_SIZE = config['batch_size']
GAMMA = config['gamma']
EPS_START = config['eps_start']
EPS_END = config['eps_end']
EPS_DECAY = config['eps_decay']
TAU = config['tau']
LR = config['lr']
MEMORY_SIZE = config['memory_size']

## STORES VALUE OF DIFFERENT PIECES ON BOARD
PAWN_VAL = 1
ROOK_VAL = 5
KNIGHT_VAL = 7
BISHOP_VAL = 3
QUEEN_VAL = 9

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))

class ReplayMemory(object):
    def __init__(self, capacity):
        self.memory = deque([], maxlen=capacity)

    def push(self, *args):
        """Save a transition"""
        self.memory.append(Transition(*args))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)

if config['load_model']:
    model_fogBot = torch.load(config['model_path'])
else:
    model_fogBot = model.FogBot()

## use epsilon greedy algorithm to decide our next state
## with probability epsilon take a random action (this enduces exploration)
## else take action to jump to state with highest estimated value
def choose_action(model, state, device, epsilon):
    values, boards = model(state)
    if random.random() < epsilon:
        idx = random.randint(0, len(values) - 1)
        return values[idx].argmax().item(), boards[idx]
    else:
        with torch.no_grad():
            choice = torch.cat(values).argmax().item()
            return choice, boards[choice]

## reward of every board is a summation of all of player's pieces subtracted by 
## the pieces of the enemies with the following weights assigned to each piece
## pawn - 1
## knight - 7
## Bishop - 3 
## rook - 5
## queen - 9 
## the king has no value as once the king is gone we have reached a game over
## and reward of the state will be either 100, or -100 depending on if victory 
## or defeat has been reached 
def get_reward(game, next_board, current_player_color): #TODO:
    white_val = 0 ## stores value of state for white player
    black_val = 0 ## stores value of state for black player
    white_king_seen = False 
    black_king_seen = False
    board_notation = next_board.board_fen() ## grabs the FEN notation string 
    ##  calculate value of board for both players
    for char in board_notation:
        if char == 'k':
            black_king_seen = True
        elif char == 'K':
            white_king_seen = True
        elif char == 'Q':
            white_val += QUEEN_VAL
        elif char == 'q':
            black_val += QUEEN_VAL
        elif char == 'R':
            white_val += ROOK_VAL
        elif char == 'r':
            black_val += ROOK_VAL
        elif char == 'B':
            white_val += BISHOP_VAL
        elif char == 'b':
            black_val += BISHOP_VAL
        elif char == 'N':
            white_val += KNIGHT_VAL
        elif char == 'n':
            black_val += KNIGHT_VAL
        elif char == 'P':
            white_val += PAWN_VAL
        elif char == 'p':
            black_val += PAWN_VAL
    
    ## if either player is missing king hardcode reward
    if(not black_king_seen):
        if(current_player_color == next_board.WHITE):
            return 100
        else:
            return -100
    if(not white_king_seen):
        if(current_player_color == next_board.BLACK):
            return 100
        else:
            return -100
    ## return appropriate reward for player 
    if(current_player_color == next_board.WHITE):
        return white_val - black_val
    else:
        return black_val - white_val
            
    

def play_self_play_game(model, replay_buffer, device, epsilon=0.3):
    game = board.CustomBoard()  # Initialize a chess game
    current_player_color = game.current_turn
    
    while not game.is_game_over():
        state = game.state.to(device)
        action, next_board = choose_action(model, state, device, epsilon) 
        next_state = next_board.state.to(device)
        reward = get_reward(game, next_board, current_player_color)
        
        done = game.is_game_over()

        replay_buffer.push(state, action, next_state, reward, done) 

        current_player_color = not current_player_color  # Switch player
        game.update_move(action)  #updates game


def dqn_train_loop(model, target_model, replay_buffer, optimizer, gamma, device):
    # ... (Model training mode, zero gradients, etc. as before) ...

    for batch_idx, (state, action, next_state, reward, done) in enumerate(data_loader):
        # ... (Compute target Q-values, current Q-values, loss) ...

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch_idx % target_update_freq == 0:
            target_model.load_state_dict(model.state_dict())


for epoch in range(num_epochs):
    for _ in range(num_self_play_games):
        play_self_play_game(model, replay_buffer, device) 

    dqn_train_loop(model, target_model, replay_buffer, optimizer, gamma, device) 
