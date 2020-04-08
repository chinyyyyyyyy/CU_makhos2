import numpy as np
from copy import deepcopy
from .preprocessing import move_to_index


class Board():


    def __init__(self, board= "" , playerTurn=1, turn=0, stale=0):
        if len(board) == 0:
            self.board = self.reset_board()
        else:
            self.board = board
            
        self.playerTurn = playerTurn
        self.turn = turn
        self.stale = stale

    
    
    def _boardPosition(self):
        bp = {-3: [], -1: [] ,1: [], 3: []}
        flatten_board = self.board.reshape(64)
        for i in range(64):
            unit = flatten_board[i]
            if unit == 0:
                continue
            else:
                bp[unit].append((i // 8,i % 8))
        return bp
    

    
    def _getCurpiece(self):
        flat_board = self.board.reshape(64)
        return flat_board[flat_board  == 0].shape[0]
        

    def _allowedActions(self):
        legal_moves = self.get_legal_moves()
        allowed_index = []
        for move in legal_moves:
            index = move_to_index(move)
            allowed_index.append(index)
        return allowed_index

    def _checkForEndGame(self):
        if self.is_game_over():
            return 1
        return 0

    def _getValue(self):
        return self.get_winner()

    def _getScore(self):
        return 0

    def _binary(self):
        board = self.board
        board = board.reshape(board.size)
        # board = np.delete(board, np.argwhere(board==5))

        currentplayer_pieces = np.zeros(board.size, dtype=np.int)
        currentplayer_kings = np.zeros(board.size, dtype=np.int)
        other_pieces = np.zeros(board.size, dtype=np.int)
        other_kings = np.zeros(board.size, dtype=np.int)

        currentplayer_pieces[board == 1] = 1
        currentplayer_kings[board == 3] = 1
        other_pieces[board == -1] = 1
        other_kings[board == -3] = 1

        return np.concatenate((currentplayer_pieces, currentplayer_kings, other_pieces, other_kings))

    def _convertStateToId(self):
        id = str(self.board)
        id = id.replace('[', '').replace(']', '').replace(
            ' ', '').replace('\n', '')
        id = id + str(self.playerTurn) + str(self.turn)
        return id

    def takeAction(self, action):
        newCheckers = deepcopy(self)
        # print(self.board)
        newCheckers.make_move(action)
        # print(newCheckers.board)
        newState = Board(newCheckers.board, -
                         self.playerTurn, newCheckers.turn)
        if newState._getCurpiece() == self._getCurpiece():
            newState.stale = self.stale+1
            # print('STALE :', newState.stale)
        return (newState, newState._getValue(), newState._checkForEndGame())

    def render(self, logger):
        logger.info('\n' + str(self.board))

    def display(self):
        tmp_board = deepcopy(self.board)
        for i in range(8):
            for j in range(8):
                if tmp_board[i][j] == -1:
                    tmp_board[i][j] = 2
                elif tmp_board[i][j] == -3:
                    tmp_board[i][j] = 4
        print(end='   ')
        for col in ['a','b','c','d','e','f','g','h']:
            print(col, end=' ')
        print()
        for row in range(8):
            print(row+1, tmp_board[row])
        print()

    # Reset the game
    def reset_board(self):
        board = np.array([
            [0, -1, 0, -1, 0, -1, 0, -1],
            [-1, 0, -1, 0, -1, 0, -1, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 1, 0, 1, 0, 1, 0, 1],
            [1, 0, 1, 0, 1, 0, 1, 0]
        ])
        self.turn = 0
        self.stale = 0
        return board

    # Check if the piece is a king

    def is_king(self, position):
        return self.board[position] == 3 or self.board[position] == -3

    # Check if a piece is an opponent

    def is_opponent(self, position):
        if self.playerTurn == 1:
            return self.board[position] == -1 or self.board[position] == -3
        elif self.playerTurn == -1:
            return self.board[position] == 1 or self.board[position] == 3

    # Switch player's turn

    def switch_player(self):
        self.playerTurn = -self.playerTurn

    # Check if the piece is in a board range
    def is_in_board_range(self, position):
        row, col = position
        return row >= 0 and row <= 7 and col >= 0 and col <= 7

    # Check if the position is occupied

    def is_occupied(self, position):
        if self.is_in_board_range(position):
            return self.board[position]

    # Get posible moves from a piece position
        
       

    def get_possible_moves(self, position):
        row, col = position
        # King
        # Search in 4 directions until found a piece or out of the board
        if self.is_king(position) and not self.is_opponent(position):
            possible_moves = []
            for i in range(7):
                nw_move = (row - i - 1, col - i - 1)
                if not self.is_in_board_range(nw_move) or self.is_occupied(nw_move):
                    break
                possible_moves.append(nw_move)
            for i in range(7):
                ne_move = (row - i - 1, col + i + 1)
                if not self.is_in_board_range(ne_move) or self.is_occupied(ne_move):
                    break
                possible_moves.append(ne_move)
            for i in range(7):
                sw_move = (row + i + 1, col - i - 1)
                if not self.is_in_board_range(sw_move) or self.is_occupied(sw_move):
                    break
                possible_moves.append(sw_move)
            for i in range(7):
                se_move = (row + i + 1, col + i + 1)
                if not self.is_in_board_range(se_move) or self.is_occupied(se_move):
                    break
                possible_moves.append(se_move)
            return possible_moves
        # Not king
        else:
             mul_player = self.playerTurn
             move = [(row - mul_player*1, col - 1),(row - mul_player*1, col + 1)]
             return [x for x in move if self.is_in_board_range(x) and not self.is_occupied(x)]
            

    # Get all possible moves from all piece
    # Return in a array of (current_position, posible_move)

    def get_all_possible_moves(self):

        #======================== Chin Part ============================
        all_possible_moves = []
        for position in self.boardPosition[self.playerTurn*1]:
            possible_moves = self.get_possible_moves(position)
            if possible_moves:
                for possible_move in possible_moves:
                            all_possible_moves.append(
                                (position, possible_move))
        for position in self.boardPosition[self.playerTurn*3]:
            possible_moves = self.get_possible_moves(position)
            if possible_moves:
                for possible_move in possible_moves:
                            all_possible_moves.append(
                                (position, possible_move))
        return all_possible_moves
    

    # Move the checker from an old position to a new position if the move is legal

    def move(self, current_position, new_position):
        possible_moves = self.get_possible_moves(current_position)
        if new_position in possible_moves:
            self.board[new_position] = self.board[current_position]
            self.board[current_position] = 0
            # print('Moved from', current_position, 'to', new_position, '\n')
            # Turn into a King
            if self.is_furthest_row(new_position):
                self.turn_into_king(new_position)
        # else:
            # print('Unable to move\n')

    # Get single jumps from a piece position
    # Return single jumps with jumped piece in a tuple of tuple ((jump_row, jump_col), [(eaten_row, eaten_col)])
    

    def get_single_jumps(self, position, first_selected_piece, selected_piece, eaten):
        row, col = position
        possible_jumps = []
        # King
        # Search in 4 directions until found a piece or out of the board
        if selected_piece == 3 or selected_piece == -3:
            for i in range(7):
                nw_jumped = (row - i - 1, col - i - 1)
                if self.is_in_board_range(nw_jumped) and self.is_occupied(nw_jumped) and nw_jumped != first_selected_piece:
                    if self.is_opponent(nw_jumped):
                        nw_jump = (row - i - 2, col - i - 2)
                        if self.is_in_board_range(nw_jump) and not self.is_occupied(nw_jump) and not nw_jumped in eaten:
                            possible_jumps.append((nw_jump, [nw_jumped]))
                        if nw_jumped in eaten:
                            continue
                    break
            for i in range(7):
                ne_jumped = (row - i - 1, col + i + 1)
                if self.is_in_board_range(ne_jumped) and self.is_occupied(ne_jumped) and ne_jumped != first_selected_piece:
                    if self.is_opponent(ne_jumped):
                        ne_jump = (row - i - 2, col + i + 2)
                        if self.is_in_board_range(ne_jump) and not self.is_occupied(ne_jump) and not ne_jumped in eaten:
                            possible_jumps.append((ne_jump, [ne_jumped]))
                        if ne_jumped in eaten:
                            continue
                    break
            for i in range(7):
                sw_jumped = (row + i + 1, col - i - 1)
                if self.is_in_board_range(sw_jumped) and self.is_occupied(sw_jumped) and sw_jumped != first_selected_piece:
                    if self.is_opponent(sw_jumped):
                        sw_jump = (row + i + 2, col - i - 2)
                        if self.is_in_board_range(sw_jump) and not self.is_occupied(sw_jump) and not sw_jumped in eaten:
                            possible_jumps.append((sw_jump, [sw_jumped]))
                        if sw_jumped in eaten:
                            continue
                    break
            for i in range(7):
                se_jumped = (row + i + 1, col + i + 1)
                if self.is_in_board_range(se_jumped) and self.is_occupied(se_jumped) and se_jumped != first_selected_piece:
                    if self.is_opponent(se_jumped):
                        se_jump = (row + i + 2, col + i + 2)
                        if self.is_in_board_range(se_jump) and not self.is_occupied(se_jump) and not se_jumped in eaten:
                            possible_jumps.append((se_jump, [se_jumped]))
                        if se_jumped in eaten:
                            continue
                    break
            
            
        # Not king
        else:
            if selected_piece == 1:
                nw_jumped = (row - 1, col - 1)
                if self.is_in_board_range(nw_jumped) and self.is_occupied(nw_jumped) and self.is_opponent(nw_jumped):
                    nw_jump = (row - 2, col - 2)
                    if self.is_in_board_range(nw_jump) and not self.is_occupied(nw_jump):
                        possible_jumps.append((nw_jump, [nw_jumped]))
                ne_jumped = (row - 1, col + 1)
                if self.is_in_board_range(ne_jumped) and self.is_occupied(ne_jumped) and self.is_opponent(ne_jumped):
                    ne_jump = (row - 2, col + 2)
                    if self.is_in_board_range(ne_jump) and not self.is_occupied(ne_jump):
                        possible_jumps.append((ne_jump, [ne_jumped]))
            else:
                sw_jumped = (row + 1, col - 1)
                if self.is_in_board_range(sw_jumped) and self.is_occupied(sw_jumped) and self.is_opponent(sw_jumped):
                    sw_jump = (row + 2, col - 2)
                    if self.is_in_board_range(sw_jump) and not self.is_occupied(sw_jump):
                        possible_jumps.append((sw_jump, [sw_jumped]))
                se_jumped = (row + 1, col + 1)
                if self.is_in_board_range(se_jumped) and self.is_occupied(se_jumped) and self.is_opponent(se_jumped):
                    se_jump = (row + 2, col + 2)
                    if self.is_in_board_range(se_jump) and not self.is_occupied(se_jump):
                        possible_jumps.append((se_jump, [se_jumped]))
        return possible_jumps

    def get_possible_jumps(self, position, first_selected_piece='', selected_piece='', eaten=[]):
        if selected_piece == '':
            selected_piece = self.board[position]
        if first_selected_piece == '':
            first_selected_piece = position
        result = []
        possible_jumps = self.get_single_jumps(
            position, first_selected_piece, selected_piece, eaten)
        for jump in possible_jumps:
            if self.get_possible_jumps(jump[0], first_selected_piece, selected_piece, eaten + jump[1]) == []:
                result += [(jump[0], eaten + jump[1])]
            result += self.get_possible_jumps(
                jump[0], first_selected_piece, selected_piece, eaten + jump[1])
        return result

    # Get all possible jumps from all piece
    # Return in a array of (current_position, posible_jump)

    def get_all_possible_jumps(self):    
    
        #======================== Chin Part ============================
        all_possible_jumps = []
        for position in self.boardPosition[self.playerTurn*1]:
            possible_jumps = self.get_possible_jumps(position)
            if possible_jumps:
                for possible_jump in possible_jumps:
                            all_possible_jumps.append(
                                (position, possible_jump[0]))
        for position in self.boardPosition[self.playerTurn*3]:
            possible_jumps = self.get_possible_jumps(position)
            if possible_jumps:
                for possible_jump in possible_jumps:
                            all_possible_jumps.append(
                                (position, possible_jump[0]))
        return all_possible_jumps
        

    # Jump from a current position to a new position if the jump is regal

    def jump(self, current_position, new_position):
        possible_jumps = self.get_possible_jumps(current_position)
        for p in possible_jumps:
            if p[0] == new_position:
                self.board[new_position] = self.board[current_position]
                self.board[current_position] = 0
                # print('Jumped from', current_position, 'to', new_position, '\n')
                for eaten in p[1]:
                    self.board[eaten] = 0
                # print(p[1], 'is removed\n')
                if self.is_furthest_row(new_position):
                    self.turn_into_king(new_position)
                break
        # else:
            # print('Unable to jump\n')

    # Check if the row is a furthest row for a current player

    def is_furthest_row(self, position):
        if self.playerTurn == 1:
            if position[0] == 0:
                return True
        elif self.playerTurn == -1:
            if position[0] == 7:
                return True
        return False

    # Turn a piece into a king if it is not a king

    def turn_into_king(self, position):
        if not self.is_king(position):
            if self.board[position] == 1:
                self.board[position] = 3
            elif self.board[position] == -1:
                self.board[position] = -3
            # print(position, 'is becomes a King\n')

    # Check if the game is over

    def is_game_over(self):
        return (len(self.get_legal_moves()) == 0) or self.is_stale()

    def is_stale(self):
        return (self.turn >= 150) or (self.stale == 32)

    # Get the winner by return
    # None if the game is not over
    # 0 if draws
    # 1 if player 1 wins
    # -1 if player 2 wins

    def get_winner(self):
        if self.is_game_over():
            if self.is_stale():
                return 0
            elif (len(self.get_legal_moves()) == 0):
                return -1
            else:
                return 1
            return 0

    def get_legal_moves(self):
        self.boardPosition = self._boardPosition()
        possible_jumps = self.get_all_possible_jumps()
        possible_moves = self.get_all_possible_moves()
        if possible_jumps:
            return possible_jumps
        elif possible_moves:
            return possible_moves
        return []

    def make_move(self, move):
        self.boardPosition = self._boardPosition()
        current_position, new_position = move
        possible_jumps = self.get_all_possible_jumps()
        possible_moves = self.get_all_possible_moves()
        self.turn += 1
        if move in possible_jumps:
            self.jump(current_position, new_position)
        elif move in possible_moves:
            self.move(current_position, new_position)
        else:
            print('invalid moves')
        self.switch_player()

    def calculate_score(self, player):
        king_weight = 1.2
        player_1 = (self.board[:, :] == 1).sum()
        player_2 = (self.board[:, :] == -1).sum()
        player_1_king = (self.board[:, :] == 3).sum()
        player_2_king = (self.board[:, :] == -3).sum()
        if player == 1:
            result = player_1_king * king_weight + player_1 - \
                player_2_king * king_weight - player_2
        elif player == -1:
            result = player_2_king * king_weight + player_2 - \
                player_1_king * king_weight - player_1
        return result

    def get_board(self):
        return self.board

    def flip(self):
    #=========== Chin Part ===================================================
    
        tempBoard = np.flipud(self.board)
        tempBoard = np.fliplr(tempBoard)
        return tempBoard * -np.ones((8,8),dtype=np.int8)

       

    def __str__(self):
        return np.array2string(self.board)

