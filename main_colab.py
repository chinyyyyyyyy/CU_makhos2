from Coach_th_checkers import Coach
from ThaiCheckers.ThaiCheckersGame import ThaiCheckersGame as Game
from utils import dotdict

import sys

"""
Before using multiprocessing, please check 2 things before use this script.
1. The number of PlayPool should not over your CPU's core number.
2. Make sure all Neural Network which each process created can store in VRAM at same time. Check your NN size before use this.
"""

args = dotdict({
    'numIters': 500,
    'numEps': 125,  # 25000
    'tempThreshold': 15,  # not used
    'updateThreshold': 0.55,  # not used
    'maxlenOfQueue': 200000,
    'numMCTSSims': 100,  # 1600 , 800
    'arenaCompare': 100,  # 400, 0
    'cpuct': 2,

    'multiGPU': False,
    'setGPU': '0',
    'numSelfPlayPool': 8,
    'numTestPlayPool': 2,

    'checkpoint': '/gdrive/My Drive/tmp_traning_data/',
    'load_model': True,
    'load_iter': str(sys.argv[1]),
    'load_folder_file': '/content/CU_makhos2/models_colab/',
    'numItersForTrainExamplesHistory': 4 , # 4
    
    
    'is_colab' : True,
    'colab_player' : str(sys.argv[2]),
    'train_params_loging' : '/content/CU_makhos2/learning_config.txt',
    'play_record_loging' : '/content/CU_makhos2/time_reports/iter',
    'test_result_logging' : '/content/CU_makhos2/results.txt',
    'models_training_logging' : '/content/CU_makhos2/loss_log.txt',
})

if __name__ == "__main__":
    g = Game()
    c = Coach(g, args)
    c.learn()

