from flask import  Flask, jsonify, request
from flask_cors import  CORS
import json

import numpy as np
import copy
import argparse
import time
import os


from utils import dotdict
from ThaiCheckers.preprocessing import index_to_move, move_to_index,index_to_move_human
from ThaiCheckers.ThaiCheckersGame import ThaiCheckersGame as Game, display
from ThaiCheckers.ThaiCheckersPlayers import minimaxAI
from ThaiCheckers.pytorch.NNet import NNetWrapper as nn
from MCTS_th_checkers import MCTS


#================= setting game ===============================================
# Argument parsers
parser = argparse.ArgumentParser('Bot select')
parser.add_argument('--type', '-t', nargs='?', dest='type', type=str)
parser.add_argument('--player2', nargs='?', dest='player2', type=bool)
parser.add_argument('--hint', nargs='?', dest='hint', type=bool, default=False)
parser.add_argument('--depth', nargs='?', dest='depth', type=int, default = 5)
parser.add_argument('--mcts', nargs='?', dest='mcts', type=int, default = 100)
args = parser.parse_args()

checkers = Game()
board = checkers.getInitBoard()


if args.type == 'minimax':
    AI = minimaxAI(checkers, depth=args.depth,verbose=True)
    print("minimax")

else:
    print('Neural network model')
    nnet = nn(checkers, gpu_num=0,use_gpu = False)
    nnet.load_checkpoint(folder='models_minimax', filename='train_iter_303.pth.tar')
    args1 = dotdict({'numMCTSSims':args.mcts, 'cpuct': 1.0})
    AI = MCTS(checkers, nnet, args1, eval=True, verbose=True)



def move_ai(board_input):
    print('Calculating...')
    valid_moves = checkers.getValidMoves(checkers.getCanonicalForm(board_input, -1), 1)
    
    if np.sum(valid_moves)==1 and args.type=='minimax':
        board, _ = checkers.getNextState(board_input, -1 , np.argmax(valid_moves))
        return
    if args.type == 'minimax':
        action = AI.get_move(checkers.getCanonicalForm(board_input, -1))
    else:
        start = time.time()
        action = np.random.choice(32*32, p=AI.getActionProb((checkers.getCanonicalForm(board_input, -1)), temp=0))
        print('Calculation Time',time.time() - start)   
    board, _ = checkers.getNextState(board_input, -1, action)
    
    return board

#================= configuration web app ===============================================

DEBUG = True

# instantiate the app
app = Flask(__name__)
app.config.from_object(__name__)

# enable CORS
CORS(app, resources={r'/*': {'origins': '*'}})




def flattern_index(idx):
    return idx[0]*8 + idx[1]

def unflattern_index(idx):
    return (idx//8,idx%8)


# get newgame request return with vector of init board
@app.route('/newgame', methods=['GET'])
def newgame():
    response_object = {'status': 'success'}
    board = checkers.getInitBoard().reshape(64).tolist()
    # board = np.array([
    #     [0, -1, 0, -1, 0, -1, 0, -1],
    #     [-1, 0, -1, 0, -1, 0, -1, 0],
    #     [0, 0, 0, 0, 0, 0, 0, 0],
    #     [0, 0, -1, 0, 0, 0, 0, 0],
    #     [0, 0, 0, 0, 0, 0, 0, 0],
    #     [0, 0, -1, 0, 0, 0, 0, 0],
    #     [0, 1, 0, 1, 0, 1, 0, 1],
    #     [1, 0, 1, 0, 1, 0, 1, 0]
    # ]).reshape(64).tolist()

    # board = np.zeros((8,8))
    # board[3,2] = -1
    # board[5,2] = 1
    # board = board.reshape(64).tolist()
    response_object['board'] = board
    return jsonify(response_object)


@app.route('/getpossiblemove', methods=['POST'])
def getpossiblemove():
    response_object = {'status': 'success'}
    curr_board = request.get_json()
    board = np.array(curr_board).reshape((8,8))
    possible_moves = []
    possible_move_idx = checkers.getValidMoves(board, 1)
    for i,idx in enumerate(possible_move_idx):
        if (idx==1):
            possible_moves.append(index_to_move(i))

    return_move = {}
    for start,end in possible_moves:
        idx = flattern_index(start)
        if idx in return_move:
            return_move[idx].append(flattern_index(end))
        else :
            return_move[idx] = [flattern_index(end)]
    response_object['possible_moves'] = return_move
    return jsonify(response_object) 

@app.route('/makemove', methods=['POST'])
def makemove():
    response_object = {'status': 'success'}

    payload = request.get_json()

    board = np.array(payload['board']).reshape((8,8))
    starting_point = unflattern_index(payload['start_pos'])
    end_point = unflattern_index(payload['end_pos'])

    board, _ = checkers.getNextState(board, 1
    , move_to_index((starting_point, end_point)))

    result = checkers.getGameEnded(board,-1)
    print(result)

    response_object['board'] =  board.reshape(64).tolist()
    response_object['result'] = result
    return jsonify(response_object)

@app.route('/aimove', methods=['POST'])
def aimove():
    response_object = {'status': 'success'}

    curr_board = request.get_json()
    board = np.array(curr_board).reshape((8,8))
    moved_board =  move_ai(board)

    result = checkers.getGameEnded(moved_board,1)
    print(result)

    response_object['board'] =  moved_board.reshape(64).tolist()
    response_object['result'] = result
    return jsonify(response_object)


# @app.route('/getgameend',methods = ['POST'])
# def getgamestate():
#     response_object = {'status': 'success'}

#     payload = request.get_json()
#     board = np.array(payload['board']).reshape((8,8))
#     player = payload['player']

#     result = checkers.getGameEnded(board,player)
#     response_object = {'result': result}
#     return jsonify(response_object)


if __name__ == '__main__':
    app.run()
