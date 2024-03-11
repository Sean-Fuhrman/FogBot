import chess

# Creat board

class CustomBoard():
    def __init__(self) -> None:
        self.create_board()

    def create_board(self):
        self.board = chess.Board()
        self.current_turn = "white" #possible values are "white" or "black"
        self.turn_number = 0


    def get_possible_moves(self): #TODO 
        return self.board.pseudo_legal_moves #DOUBLE CHECK THIS RETURNS PROPER FOG of WAR CHESS legal moves
    
    def is_game_over(self): #TODO
        return self.board.is_game_over() # Check if both kings are present on board

    def update_move(self, move): #TODO: Update board representation with mvoes 
        self.turn_number += 1
        self.current_turn = "white" if self.current_turn == "black" else "black"

        #move is output of model

    def get_board_state(self): #TODO: return board as embedding
        pass

    def get_next_move_states(self); #TODO: return all next move state embeddings.
        pass