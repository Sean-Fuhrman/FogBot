import numpy as np

import torch as torch 
import torch.nn as nn
import torch.nn.functional as F
import board
import model
import chess
import math
import random
import time
import matplotlib.pyplot as plt

from collections import namedtuple, deque
from itertools import count

import yaml

steps_done = 0


Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'mask', 'next_mask','reward', 'not_done'))

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
    
def get_legal_move_mask(game):
    mask = torch.zeros(64*64)
    valid_moves = game.get_possible_moves()
    for move in valid_moves:
        from_square = move.from_square
        to_square = move.to_square
        mask[from_square*64 + to_square] = 1

    return mask.bool()
def convert_action_to_move(action, game):
    return chess.Move(from_square=action//64, to_square=action%64)
## use epsilon greedy algorithm to decide our next state
## with probability epsilon take a random action (this enduces exploration)
## else take action to jump to state with highest estimated value
def choose_action(model, state,game, device,config):
    with torch.no_grad():
        global steps_done

        epsilon_start = config['eps_start']
        epsilon_end = config['eps_end']
        epsilon_decay = config['eps_decay']
        epsilon = epsilon_end + (epsilon_start - epsilon_end) * math.exp(-1. * steps_done / epsilon_decay)
        steps_done += 1
        values = model(state)
        mask = get_legal_move_mask(game)
        values[~mask] = -float("inf") # Set illegal moves to -inf so they are not chosen
        if random.random() < epsilon:
            return torch.tensor([random.choice(mask.nonzero(as_tuple=True)[0])]).to(device), mask
        else:
            selected_index = torch.multinomial(F.softmax(values, dim=0), 1)
            return selected_index.to(device), mask


def get_white_score(game):
    piece_values = {
        chess.PAWN: 1,
        chess.KNIGHT: 3,
        chess.BISHOP: 3,
        chess.ROOK: 5,
        chess.QUEEN: 9,
        chess.KING: 0  # King's value is often excluded here
    }

    white_val = 0 ## stores value of state for white player
    black_val = 0 ## stores value of state for black player

    for square in chess.SQUARES:
        piece = game.board.piece_at(square)
        if piece and piece.color == chess.WHITE:
            white_val += piece_values[piece.piece_type]
        elif piece and piece.color == chess.BLACK:
            black_val += piece_values[piece.piece_type]

    return white_val - black_val

def get_reward(game, next_board, current_player_color): #TODO:
    white_king_seen = next_board.board.king(chess.WHITE) != None
    black_king_seen = next_board.board.king(chess.BLACK) != None
    
    ## if either player king is missing king hardcode reward
    if(not black_king_seen):
        if(current_player_color == chess.WHITE):
            return 100
        else:
            return -100
    if(not white_king_seen):
        if(current_player_color == chess.BLACK):
            return 100
        else:
            return -100
    # print(game.board.fen() == next_board.board.fen())
    prev_val = get_white_score(game)
    next_val = get_white_score(next_board)
    # print(f"prev_val: {prev_val}, next_val: {next_val}")
    if(current_player_color == chess.WHITE):
        return next_val - prev_val
    else:
        return prev_val - next_val

def make_target_net_move(game, target_net, device):
    state = game.get_board_state().to(device)
    values = target_net(state).to(device)
    mask = get_legal_move_mask(game)
    values = values * mask.to(device)
    action = torch.argmax(values)
    game.update_move(convert_action_to_move(action.to("cpu"), game))
    return game

def train_loop(policy_net, target_net, replay_buffer, config, device):
    game = board.CustomBoard(device)  # Initialize a chess game
    policy_color = random.choice([chess.WHITE, chess.BLACK])
    policy_color_string = "white" if policy_color == chess.WHITE else "black"
    print(f"Policy color: {policy_color_string}")
    if policy_color == chess.BLACK:
        game = make_target_net_move(game, target_net, device)
    losses = []
    reward_sum = 0
    while not game.is_game_over():
        state = game.get_board_state().to(device)
        action, mask = choose_action(policy_net,state, game, device, config) 
        
        next_board = game.copy()
        next_board.update_move(convert_action_to_move(action))

        if not next_board.is_game_over():
            next_board = make_target_net_move(next_board, target_net, device)       

        reward = torch.tensor([get_reward(game, next_board, policy_color)])
        done = game.is_game_over()
        reward_sum += reward.item()
        next_state = next_board.get_board_state().to(device)
        next_mask = get_legal_move_mask(next_board)

        replay_buffer.push(state, action, next_state,mask, next_mask, reward, not done) 
        loss = optimize_model(policy_net, target_net, replay_buffer, optimizer, config, device)  # Optimize the model
        if loss != -1:
            losses.append(loss)
        game = next_board  # Update the game state
        
        # Soft update of the target network's weights
        # θ′ ← τ θ + (1 −τ )θ′
        TAU = config['tau']
        target_net_state_dict = target_net.state_dict()
        policy_net_state_dict = policy_net.state_dict()
        for key in policy_net_state_dict:
            target_net_state_dict[key] = policy_net_state_dict[key]*TAU + target_net_state_dict[key]*(1-TAU)
        target_net.load_state_dict(target_net_state_dict)
    return torch.mean(torch.tensor(losses)).item(), reward_sum

#gotten from https://pytorch.org/tutorials/intermediate/reinforcement_q_learning.html
def optimize_model(policy_net, target_net, replay_buffer, optimizer, config, device):
    BATCH_SIZE = config['batch_size']
    GAMMA = config['gamma']
    if len(replay_buffer) < config['batch_size']:
        return -1

    transitions = replay_buffer.sample(config['batch_size'])
    batch = Transition(*zip(*transitions))


    # Compute a mask of non-final states and concatenate the batch elements
    # (a final state would've been the one after which simulation ended)
    non_final_mask = torch.tensor(batch.not_done).to(device)
    non_final_next_states = torch.stack(batch.next_state)[non_final_mask].to(device)
    mask_batch = torch.stack(batch.mask).to(device)
    next_mask_batch = torch.stack(batch.next_mask).to(device)
    state_batch = torch.stack(batch.state).to(device)
    action_batch = torch.cat(batch.action).to(device)
    reward_batch = torch.cat(batch.reward).to(device)

    # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
    # columns of actions taken. These are the actions which would've been taken
    # for each batch state according to policy_net
    state_action_values =torch.gather(policy_net(state_batch), dim=1, index=action_batch.unsqueeze(1))

    # Compute V(s_{t+1}) for all next states.
    # Expected values of actions for non_final_next_states are computed based
    # on the "older" target_net; selecting their best reward with max(1).values
    # This is merged based on the mask, such that we'll have either the expected
    # state value or 0 in case the state was final.
    next_state_values = torch.zeros(BATCH_SIZE, device=device)
    with torch.no_grad():
        next_state_values[non_final_mask] = (target_net(non_final_next_states) * next_mask_batch[non_final_mask].int()).max(1).values
    # Compute the expected Q values
    expected_state_action_values = (next_state_values * GAMMA) + reward_batch

    # Compute Huber loss
    criterion = nn.SmoothL1Loss()
    loss = criterion(state_action_values, expected_state_action_values.unsqueeze(1))
    # Optimize the model
    optimizer.zero_grad()
    loss.backward()
    # In-place gradient clipping
    torch.nn.utils.clip_grad_value_(policy_net.parameters(), 100)
    optimizer.step()
    return loss.item()


if __name__ == "__main__":
    config = None
    device = "cpu"
    if torch.cuda.is_available():
        device = "cuda"
    if torch.backends.mps.is_available():
        device = "mps"
        
    with open('config.yaml', 'r') as file:
        config = yaml.safe_load(file)

    num_games = config['num_games']
    replay_buffer = ReplayMemory(config['replay_buffer_size'])
    policy_net = model.DQN()
    if config['load_model']:
        policy_net = torch.load(config['model_path'])
    policy_net = policy_net.to(device)
    target_net = model.DQN().to(device)
    target_net.load_state_dict(policy_net.state_dict())

    optimizer = torch.optim.Adam(policy_net.parameters(), lr=config['lr'])

    losses=[]

    for game_index in range(num_games):
        print(f"Starting game {game_index}")
        time_start = time.time()
        avg_loss, reward_sum = train_loop(policy_net, target_net, replay_buffer, config, device) 
        losses.append(avg_loss)
        time_end = time.time()
        print(f"Game {game_index} over, with average loss {avg_loss}, and {reward_sum} total reward, took {time_end - time_start} seconds.")
        if game_index % 100 == 0:
            torch.save(target_net, "models/" + config['model_path'])
            plt.plot(losses)
            plt.savefig("plots/losses.png")
            plt.close()
            print("Model saved")

