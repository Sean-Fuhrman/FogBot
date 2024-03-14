import chess
import board
import torch
import model
import yaml
import train
#load model
fog_bot = model.DQN()

config = None
with open('config.yaml', 'r') as file:
    config = yaml.safe_load(file)

fog_bot = torch.load(config['model_path'])

game = board.CustomBoard("cpu")

def select_action(board, fogBot):
    state = board.generate_state()
    values = fogBot(state)
    mask = train.get_legal_move_mask(board)
    values = values * mask
    action = torch.argmax(values)
    move = train.convert_action_to_move(action)
    return move

print(game.board)
while(not game.is_game_over()):

    print("Enter move: ")
    move = input()
    game.push(move)
    print(game.board)
    print("Bot move: ")
    bot_move = select_action(game, fog_bot)
    game.push(bot_move)
    print(game.board)