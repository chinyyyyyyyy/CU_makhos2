from Coach_th_checkers import Coach
from ThaiCheckers.ThaiCheckersGame import ThaiCheckersGame as Game
from ThaiCheckers.pytorch.NNet import NNetWrapper as nn
from utils import *
import os

"""
Before using multiprocessing, please check 2 things before use this script.
1. The number of PlayPool should not over your CPU's core number.
2. Make sure all Neural Network which each process created can store in VRAM at same time. Check your NN size before use this.
"""

args = dotdict({
    'numIters': 500,
    'numEps': 500,  # 25000
    'tempThreshold': 15,  # not used
    'updateThreshold': 0.55,  # not used
    'maxlenOfQueue': 200000,
    'numMCTSSims': 100,  # 1600 , 800
    'arenaCompare': 50,  # 400, 0
    'cpuct': 2,

    'multiGPU': False,
    'setGPU': '1',
    'numSelfPlayPool': 12,
    'numTestPlayPool': 12,

    'checkpoint': '/root/test/CU_Makhos/models_minimax/',
    'load_model': True,
    'load_iter': 30,
    'load_folder_file': '/root/test/CU_Makhos/models_minimax/',
    'numItersForTrainExamplesHistory': 4 , # 4
    
    'shared_tree' : True

})

if __name__ == "__main__":

    os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"
    g = Game()
    c = Coach(g, args)
    # c.nnet1.load_checkpoint(folder=args.checkpoint, filename='train_iter_'+str(args.load_iter)+'.pth.tar')
    # c.parallel_self_test_play(1)
    # c.learn_minimax()
    # c.args.load_iter = 30
    c.learn()
