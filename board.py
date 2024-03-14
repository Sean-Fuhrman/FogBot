import chess
import torch
import copy
# Creat board

class CustomBoard():
    def __init__(self, device, fen=None) -> None:
        self.device = device
        self.create_board()
        self.generate_state()
        if fen is not None:
            self.load_string(fen)

    def create_board(self):
        self.board = chess.Board()
        self.current_turn = chess.WHITE
        self.turn_number = 0

    def generate_state(self):
        #empty, pawn, knight, bishop, rook, queen, king, is_owned , fog 
        self.one_hot = torch.zeros(8,8,9)
        for i in range(8):
            for j in range(8):
                piece = self.board.piece_at(chess.square(i,j))
                color = self.board.color_at(chess.square(i,j))
                fogged = self.is_square_fogged(chess.square(i,j))
                if fogged:
                    self.one_hot[i,j,8] = 1
                elif piece:
                    self.one_hot[i,j,piece.piece_type] = 1
                    if color == self.current_turn:
                        self.one_hot[i,j,7] = 1
                else:
                    self.one_hot[i,j,0] = 1
            

        self.state = self.one_hot
        self.state = torch.cat((self.state.flatten(), torch.tensor([self.turn_number]), torch.tensor([self.current_turn])), dim=0)

    def is_square_fogged(self, square):
        if square in [x.to_square for x in self.board.pseudo_legal_moves]:
            return False
        elif self.board.color_at(square) == self.current_turn:
            return False
        else:
            return True

    def get_possible_moves(self): 
        return self.board.pseudo_legal_moves
    
    def board_to_string(self):
        return self.board.fen()

    def load_string(self, string):
        self.board = chess.Board(string)
        self.generate_state()
    
    def is_game_over(self):
        if self.board.king(chess.WHITE) == None or self.board.king(chess.BLACK) == None:
            return True
        return False

    def update_move(self, move):
        if(self.current_turn == chess.BLACK):
            self.turn_number += 1
            self.current_turn = chess.WHITE
        else:
            self.current_turn = chess.BLACK
        self.board.push(move)
        self.generate_state()

    def get_board_state(self):
        return self.state.to(self.device)

    def copy(self):
        return copy.deepcopy(self)