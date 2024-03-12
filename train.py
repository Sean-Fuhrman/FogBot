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

def choose_action(model, state, device, epsilon):
    values, boards = model(state)
    if random.random() < epsilon:
        idx = random.randint(0, len(values) - 1)
        return values[idx].argmax().item(), boards[idx]
    else:
        with torch.no_grad():
            choice = torch.cat(values).argmax().item()
            return choice, boards[choice]

def get_reward(game, next_board, current_player_color): #TODO:
    return 0  

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
