#%%
import board
import chess
new_board = board.CustomBoard()

new_board.board
# %%
print(new_board.evaluate_relative_to(chess.WHITE))
