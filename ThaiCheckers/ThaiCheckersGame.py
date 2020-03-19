import numpy as np
from .ThaiCheckersLogic import Board
from Game import Game
from .preprocessing import index_to_move
import sys
sys.path.append('..')


class ThaiCheckersGame(Game):
    def __init__(self, board=[], playerTurn=1, turn=0):
        Game.__init__(self)

        self.currentPlayer = 1
        self.gameState = Board([], 1, 0, 0)  # board class
        self.actionSpace = np.array([0] * 32 * 32)
        self.pieces = {'-3': '-3', '-1': '-1',
                       '0': '0', '1': '1', '3': '3'}
        self.grid_shape = (8, 8)
        self.input_shape = (4, 8, 8)
        self.name = 'checkers'
        self.action_size = len(self.actionSpace)

    def getInitBoard(self):
        # return initial board (numpy board)
        return self.gameState.reset_board()

    def getBoardSize(self):
        # (a,b) tuple
        return self.grid_shape

    def getActionSize(self):
        # return number of actions
        return self.action_size

    def getNextState(self, board, player, action):
        canonicalBoard = self.getCanonicalForm(board, player)

        a = index_to_move(action)
        self.gameState = Board(
            canonicalBoard, 1, self.gameState.turn, self.gameState.stale)
        next_state, _, _ = self.gameState.takeAction(a)
        self.gameState = next_state
        self.currentPlayer = -player

        if (player == self.gameState.PLAYER_2):
            self.gameState.board = self.gameState.flip()

        return self.gameState.board, -player

    def getValidMoves(self, board, player):
        # return a fixed size binary vector
        b = Board(board, player)
        valids = self.actionSpace.copy()
        for idx in b._allowedActions():
            valids[idx] = 1
        return valids

    def getGameEnded(self, board, player):
        # return 0 if not ended, 1 if player 1 won, -1 if player 1 lost
        # player = 1
        canonicalBoard = self.getCanonicalForm(board, player)
        b = Board(canonicalBoard, 1,
                  self.gameState.turn, self.gameState.stale)
        # print('check game end')
        if b.is_game_over():
            if b.get_winner() == 0:
                return 1e-4
            return b.get_winner() * player
        else:
            return 0

    def getCanonicalForm(self, board, player):
        if player == self.gameState.PLAYER_2:
            return Board(board.copy()).flip()
        return board

    def getSymmetries(self, board, pi):
        # no symmetric in checkers
        return [(board, pi)]

    def stringRepresentation(self, boardHistory):
        # 8x8 numpy array (canonical board)
        
        out = bytes()
        for i in range(len(boardHistory)):
            out += boardHistory[i].tobytes()
        return out + bytes([self.gameState.turn]) + bytes([self.gameState.stale])


