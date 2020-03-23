from collections import deque
import os
import logging
import random
from random import shuffle
import time
import sys

import numpy as np
import pandas as pd
import pickle
from torch import multiprocessing
import torch
import psutil

from MCTS_th_checkers import MCTS
from Arena import Arena
from ThaiCheckers.pytorch.NNet import NNetWrapper as nn
from ThaiCheckers.ThaiCheckersPlayers import minimaxAI
from utils_examples_global_avg import build_unique_examples
from utils import dotdict
import shutil



mp = multiprocessing.get_context('spawn')

logging.basicConfig(
    level=logging.DEBUG,
    format='[%(asctime)s %(processName)s]\t %(message)s'
    )


def AsyncSelfPlay(merge_tree, nnet, game, args, iter_num, ns): 

    mcts = MCTS(game, nnet, args, merge_tree)    
    ##================ Memory Freze protection ====================
    
    if ns.leak:
        print('memory leak already leak')
        return
   
    if (psutil.virtual_memory()[2]) > 90:
        print('memory leak')
        ns.leak = True
        return 
    ##=============================================================
    
    logging.debug("self playing game" + str(iter_num))

    os.environ["CUDA_VISIBLE_DEVICES"] = args.setGPU
    start_game_time = time.time() 

    
    trainExamples = []
    board = game.getInitBoard()
    curPlayer = 1
    episodeStep = 0
    moves_records = []

    while True:
        start_move = time.time()
        episodeStep += 1
        canonicalBoard = game.getCanonicalForm(board, curPlayer)
        pi = mcts.getActionProb(canonicalBoard, temp=1)
    
        
        valids = game.getValidMoves(canonicalBoard, 1)

        trainExamples.append([canonicalBoard, curPlayer, pi,
                              game.gameState.turn, game.gameState.stale, valids])

        action = random.choices(np.arange(0, len(pi)), weights=pi)[0]
        board, curPlayer = game.getNextState(
            board, curPlayer, action)

 
        r = game.getGameEnded(board, curPlayer)  # winner
        moves_records.append(time.time() - start_move)
        

        if r != 0:
            end_game_time  = time.time()
            game_duration = end_game_time - start_game_time
            p = psutil.Process()
            report = [iter_num, start_game_time, end_game_time, game_duration, p.cpu_num(), p.memory_info()[0]/(1024*1024*1024), moves_records]
            logging.debug("side of Ps " + str(sys.getsizeof(mcts.Ps)/1024))
            logging.debug("side of Es " + str(sys.getsizeof(mcts.Es)/1024))
            logging.debug("side of Vs " + str(sys.getsizeof(mcts.Vs)/1024))
            return [(x[0], x[2], r*x[1], x[3], x[4], x[5]) for x in trainExamples], r, report ,mcts.Ps, mcts.Es, mcts.Vs


def AsyncMinimaxPlay(game, args,gameth):
    
    logging.debug("play minimax game " + str(gameth))
    minimax = minimaxAI(game)
    trainExamples = []
    board = game.getInitBoard()
    curPlayer = 1
    episodeStep = 0

    while True:
        episodeStep += 1
        canonicalBoard = game.getCanonicalForm(board, curPlayer)
        # boardHistory.append(canonicalBoard)

        pi = minimax.get_pi(canonicalBoard)
        valids = game.getValidMoves(canonicalBoard, 1)

        trainExamples.append([canonicalBoard, curPlayer, pi,
                              game.gameState.turn, game.gameState.stale, valids])

        action = random.choices(np.arange(0, len(pi)), weights=pi)[0]
        board, curPlayer = game.getNextState(
            board, curPlayer, action)

        r = game.getGameEnded(board, curPlayer)  # winner

        if r != 0:

            return [(x[0], x[2], r*x[1], x[3], x[4], x[5]) for x in trainExamples], r 


def TrainNetwork(nnet, game, args, iter_num, trainhistory, train_net=True):
    # ---load history file---
    
    modelFile = os.path.join(args.checkpoint, "trainhistory.pth.tar")
    examplesFile = modelFile+".examples"
    if not os.path.isfile(examplesFile):
        print('Train history not found')
    else:
        #make a backup
        shutil.copy2(examplesFile, examplesFile+'.tmp')
        # print("File with trainExamples found. Read it.")
        old_history = pickle.load(open(examplesFile, "rb"))
        for iter_samples in old_history:
            trainhistory.append(iter_samples)
        # f.closed
    # ----------------------
    # ---delete if over limit---
    if len(trainhistory) > args.numItersForTrainExamplesHistory:
        print("len(trainExamplesHistory) =", len(trainhistory),
              " => remove the oldest trainExamples")
        #del trainhistory[len(trainhistory)-1]
        trainhistory = trainhistory[:args.numItersForTrainExamplesHistory]
        print('Length after remove:', len(trainhistory))

    # ---save history---
    folder = args.checkpoint
    if not os.path.exists(folder):
        os.makedirs(folder)
    filename = os.path.join(folder, 'trainhistory.pth.tar'+".examples")
    pickle.dump(trainhistory, open(filename, "wb"))

    # with open(filename, "wb") as f:
    #     Pickler(f).dump(trainhistory)
    #     # f.closed
    # ------------------
    if train_net:
        trainExamples = build_unique_examples(trainhistory)
        shuffle(trainExamples)
        print('Total train samples (moves):', len(trainExamples))

        nnet.train(trainExamples)
        
        
        if args.shared_tree:
            nnet.save_checkpoint(folder=args.checkpoint,
                            filename='train_iter_shared_tree_' + str(iter_num) + '.pth.tar')
        else:
            nnet.save_checkpoint(folder=args.checkpoint,
                            filename='train_iter_' + str(iter_num) + '.pth.tar')


def AsyncAgainst(nnet, game, args, gameth):
    
    logging.debug("play self test game " + str(gameth))


    os.environ["CUDA_VISIBLE_DEVICES"] = args.setGPU

    # create nn and load
    minimax = minimaxAI(game)

    local_args = dotdict({'numMCTSSims': 100, 'cpuct': 1.0})
    # local_args.numMCTSSims = 100
    # local_args.cpuct = 1
    mcts = MCTS(game, nnet, local_args, eval=True)

    arena = Arena(lambda x: np.argmax(mcts.getActionProb(x, temp=0)),
                  minimax.get_move, game)
    arena.displayBar = False
    net_win, minimax_win, draws = arena.playGames(2)
    return net_win, minimax_win, draws


class Coach():
    """
    This class executes the self-play + learning. It uses the functions defined
    in Game and NeuralNet. args are specified in main.py.
    """

    def __init__(self, game, args):
        self.game = game
        self.args = args
        self.nnet1 = nn(self.game, gpu_num=self.args.setGPU)


        self.trainExamplesHistory = []
        self.checkpoint_iter = 0

        self.win_count = 0
        self.loss_count = 0
        self.draw_count = 0

        self.win_games = []
        self.loss_games = []
        self.draw_games = []

    def parallel_self_play(self):
        

        # temp_draw_games = []
        # temp_win_games = []
        # temp_loss_games = []
        
        reports = []
        
        mana = multiprocessing.Manager()
        ns = mana.Namespace()
        ns.leak = False
        
        merge_tree = {0 : {}, 1 : {}, 2 : {}}
        
        
        for k in range(5):
            pool = mp.Pool(processes=self.args.numSelfPlayPool, maxtasksperchild=1)
            res = []
           
            print(psutil.virtual_memory()[2])

            for i in range(5):
                net = self.nnet1
                res.append(pool.apply_async(AsyncSelfPlay, args=(
                    merge_tree, net, self.game, self.args, i,ns)))
                print(psutil.virtual_memory()[2])
    
            pool.close()
            pool.join()
            pool.terminate()
            
            if ns.leak:
                pool.terminate()
                print("terminate program")
                sys.exit()
                
            Ps_tree =  {0:{}, 1:{}, 2:{}, 3:{}, 4:{}, 5:{}, 6:{}, 7:{}, 8:{}, 9:{}}
            Es_tree =  {0:{}, 1:{}, 2:{}, 3:{}, 4:{}, 5:{}, 6:{}, 7:{}, 8:{}, 9:{}}
            Vs_tree =  {0:{}, 1:{}, 2:{}, 3:{}, 4:{}, 5:{}, 6:{}, 7:{}, 8:{}, 9:{}}
        
            
            logging.debug("finish learning merging tree")
            
    
            for i in range(len(res)):
                gameplay, r, report, Ps_tree[i], Es_tree[i], Vs_tree[i] = res[i].get()
                reports.append(report)
                if (r == 1e-4):
                    self.draw_count += 1
                    self.draw_games.append(gameplay)
                elif r == 1:
                    self.win_count += 1
                    self.win_games.append(gameplay)
                else:
                    self.loss_count += 1
                    self.loss_games.append(gameplay)
    

            # for i in temp_draw_games:
            #     self.draw_games += i
    
            # for i in temp_win_games:
            #     self.win_games += i
    
            # for i in temp_loss_games:
            #     self.loss_games += i
                
                
            s = time.time()
                
            merge_ps_tree = {**Ps_tree[0], **Ps_tree[1], **Ps_tree[2], **Ps_tree[3], **Ps_tree[4], **Ps_tree[5], **Ps_tree[6], **Ps_tree[7], **Ps_tree[8], **Ps_tree[9]}
            merge_vs_tree = {**Vs_tree[0], **Vs_tree[1], **Vs_tree[2], **Vs_tree[3], **Vs_tree[4], **Vs_tree[5], **Vs_tree[6], **Vs_tree[7], **Vs_tree[8], **Vs_tree[9]}
            merge_es_tree = {**Es_tree[0], **Es_tree[1], **Es_tree[2], **Es_tree[3], **Es_tree[4], **Es_tree[5], **Es_tree[6], **Es_tree[7], **Es_tree[8], **Es_tree[9]}
            
            merege_time = time.time() - s
            
            print(sys.getsizeof(merge_ps_tree)/1024)
            print(sys.getsizeof(merge_vs_tree)/1024)
            print(sys.getsizeof(merge_es_tree)/1024)
            print("merge time =",merege_time)
            
            merge_tree[0] = merge_ps_tree
            merge_tree[1] = merge_es_tree
            merge_tree[2] = merge_vs_tree
        
        
        
        
        return reports

    def parallel_minimax_play(self):
        pool = mp.Pool(processes=self.args.numSelfPlayPool)
        temp = []
        res = []
        result = []

        temp_draw_games = []
        temp_win_games = []
        temp_loss_games = []
        for i in range(self.args.numEps):

            res.append(pool.apply_async(AsyncMinimaxPlay, args=(
                self.game, self.args,i)))

        pool.close()
        pool.join()

        for i in res:
            gameplay, r = i.get()
            result.append(gameplay)
            if (r == 1e-4):
                self.draw_count += 1
                temp_draw_games.append(gameplay)
            elif r == 1:
                self.win_count += 1
                temp_win_games.append(gameplay)
            else:
                self.loss_count += 1
                temp_loss_games.append(gameplay)

        for i in result:
            temp += i
        for i in temp_draw_games:
            self.draw_games += i

        for i in temp_win_games:
            self.win_games += i

        for i in temp_loss_games:
            self.loss_games += i
        return temp

    def parallel_self_test_play(self, iter_num):
        print("start self play ",iter_num)
        pool = mp.Pool(processes=self.args.numTestPlayPool, maxtasksperchild=1)

        res = []
        result = []
        net = self.nnet1
        for i in range(self.args.arenaCompare):
            res.append(pool.apply_async(
                AsyncAgainst, args=(net, self.game, self.args, i)))
        pool.close()
        pool.join()

        pwins = 0
        nwins = 0
        draws = 0
        for i in res:
            result.append(i.get())
        for i in result:
            pwins += i[0]
            nwins += i[1]
            draws += i[2]

        out = "iter "+str(iter_num)+"\tNN win: "+str(pwins)+"\tMinimax win: " + str(nwins)+"\tDraws: "+str(draws)
        print(out)
        f = open('/root/test/CU_Makhos/results.txt','a')
        f.write(out+"\n")
        f.close()
 

    def train_network(self, iter_num, train_net=True):

        print("Start train network")

        torch.cuda.set_device('cuda:' + self.args.setGPU)

        TrainNetwork(self.nnet1, self.game, self.args,
                     iter_num, self.trainExamplesHistory, train_net)

    def learn(self):
        """
        Performs numIters iterations with numEps episodes of self-play in each
        iteration. After every iteration, it retrains neural network with
        examples in trainExamples (which has a maximium length of maxlenofQueue).
        It then pits the new neural network against the old one and accepts it
        only if it wins >= updateThreshold fraction of games.
        """

        if self.args.load_model:
            try:
                self.nnet1.load_checkpoint(
                    folder=self.args.checkpoint, filename='train_iter_'+str(self.args.load_iter)+'.pth.tar')
            
        
            except Exception as e:
                print(e)
                print("Create a new model")

        pytorch_total_params = sum(p.numel()
                                   for p in self.nnet1.nnet.parameters() if p.requires_grad)

        print('Num trainable params:', pytorch_total_params)

        print('LR:')
        for param_group in self.nnet1.optimizer.param_groups:
            print(param_group['lr'])

        

        start_iter = 1
        if self.args.load_model:
            start_iter += self.args.load_iter
            self.args.numMCTSSims += self.args.load_iter
            self.args.numItersForTrainExamplesHistory = min(
                20, 4 + (self.args.load_iter-4)//2)

        for i in range(start_iter, self.args.numIters+1):
            if (self.args.numMCTSSims < 400):
                self.args.numMCTSSims += 1
            if ((i > 5) and (i % 2 == 0) and (self.args.numItersForTrainExamplesHistory < 20)):
                self.args.numItersForTrainExamplesHistory += 1
            
            learning_config = '------ITER ' + str(i) + '------' + '\tMCTS sim:' + str(self.args.numMCTSSims) + '\tIter samples :' + str(self.args.numItersForTrainExamplesHistory)
            print(learning_config)
            f = open('/root/test/CU_Makhos/learning_config.txt','a')
            f.write(learning_config + "\n")
            f.close()
            
            self.win_count = 0
            self.loss_count = 0
            self.draw_count = 0

            self.win_games = []
            self.loss_games = []
            self.draw_games = []

            iterationTrainExamples = deque([], maxlen=self.args.maxlenOfQueue)

            reports = self.parallel_self_play()
            report_df = pd.DataFrame(reports)
            report_df.to_csv('/root/test/CU_Makhos/time_reports/iter' + str(i))
            sys.exit()
            
            

            # iterationTrainExamples += temp
            iterationTrainExamples += self.win_games
            iterationTrainExamples += self.loss_games

            print('Win count:', self.win_count, 'Loss count:',
                  self.loss_count, 'Draw count:', self.draw_count)

            self.checkpoint_iter = i

    

            if self.draw_count <= (self.win_count + self.loss_count):
                iterationTrainExamples += self.draw_games

            else:
                win_loss_count = len(self.win_games) + len(self.loss_games)

                sample_draw_games = random.sample(
                    self.draw_games, win_loss_count)  # get samples from draw games

                iterationTrainExamples += sample_draw_games
                print('Too much draw, add all win/loss games and ',
                      str(win_loss_count), ' draw moves')

            self.trainExamplesHistory.append(iterationTrainExamples)
            self.train_network(i)
            self.trainExamplesHistory.clear()

            if i % 10 == 0:
                self.parallel_self_test_play(i)


    def learn_minimax(self):

        if self.args.load_model:
            try:
                self.nnet1.load_checkpoint(
                    folder=self.args.checkpoint, filename='train_iter_'+str(self.args.load_iter)+'.pth.tar')
                # self.nnet1.load_state_dict(self.nnet.state_dict())
                # self.nnet2.load_state_dict(self.nnet.state_dict())

            except Exception as e:
                print(e)
                print("Create a new model")

        pytorch_total_params = sum(p.numel()
                                   for p in self.nnet1.nnet.parameters() if p.requires_grad)

        print('Num trainable params:', pytorch_total_params)

        # start_iter = 1
        # if self.args.load_model:
        #     start_iter += self.args.load_iter

        for i in range(2, 30):  # hard code for 30 iters
            print('------ITER ' + str(i) + '------')
            self.win_count = 0
            self.loss_count = 0
            self.draw_count = 0

            self.win_games = []
            self.loss_games = []
            self.draw_games = []

            iterationTrainExamples = deque([], maxlen=self.args.maxlenOfQueue)

            self.parallel_minimax_play()

            # iterationTrainExamples += temp
            iterationTrainExamples += self.win_games
            iterationTrainExamples += self.loss_games
            iterationTrainExamples += self.draw_games

            print('Win count:', self.win_count, 'Loss count:',
                  self.loss_count, 'Draw count:', self.draw_count)

            self.checkpoint_iter = i

            self.trainExamplesHistory.append(iterationTrainExamples)
            self.train_network(i)
            self.parallel_self_test_play(i)
            self.trainExamplesHistory.clear()


    def learn_rerun(self):
        if self.args.load_model:
            try:
                self.nnet1.load_checkpoint(
                    folder=self.args.checkpoint, filename='train_iter_'+str(self.args.load_iter)+'.pth.tar')
                # self.nnet1.load_state_dict(self.nnet.state_dict())
                # self.nnet2.load_state_dict(self.nnet.state_dict())

            except Exception as e:
                print(e)
                print("Create a new model")

        pytorch_total_params = sum(p.numel()
                                   for p in self.nnet1.nnet.parameters() if p.requires_grad)

        print('Num trainable params:', pytorch_total_params)

        print('LR:')
        for param_group in self.nnet1.optimizer.param_groups:
            print(param_group['lr'])

        #state_dict = self.nnet1.nnet.state_dict()
        # self.nnet1.nnet.load_state_dict(state_dict)
        #self.nnet2.nnet.load_state_dict(state_dict)
        # self.nnet3.nnet.load_state_dict(state_dict)

        start_iter = 1
        if self.args.load_model:
            start_iter += self.args.load_iter
            self.args.numMCTSSims += self.args.load_iter
            self.args.numItersForTrainExamplesHistory = min(
                20, 4 + (self.args.load_iter-4)//2)

        for i in range(start_iter, start_iter+1):
            if (self.args.numMCTSSims < 400):
                self.args.numMCTSSims += 1
            if ((i > 5) and (i % 2 == 0) and (self.args.numItersForTrainExamplesHistory < 20)):
                self.args.numItersForTrainExamplesHistory += 1
            print('------ITER ' + str(i) + '------' +
                  '\tMCTS sim:' + str(self.args.numMCTSSims) + '\tIter samples :' + str(self.args.numItersForTrainExamplesHistory))

            self.win_count = 0
            self.loss_count = 0
            self.draw_count = 0

            self.win_games = []
            self.loss_games = []
            self.draw_games = []

            iterationTrainExamples = deque([], maxlen=self.args.maxlenOfQueue)

            self.parallel_self_play()

            # iterationTrainExamples += temp
            iterationTrainExamples += self.win_games
            iterationTrainExamples += self.loss_games

            print('Win count:', self.win_count, 'Loss count:',
                  self.loss_count, 'Draw count:', self.draw_count)

            self.checkpoint_iter = i

            # games = []
            # games += self.win_games
            # games += self.loss_games

            if self.draw_count <= (self.win_count + self.loss_count):
                iterationTrainExamples += self.draw_games

            else:
                win_loss_count = len(self.win_games) + len(self.loss_games)

                sample_draw_games = random.sample(
                    self.draw_games, win_loss_count)  # get samples from draw games

                iterationTrainExamples += sample_draw_games
                print('Too much draw, add all win/loss games and ',
                      str(win_loss_count), ' draw moves')

            self.trainExamplesHistory.append(iterationTrainExamples)
            self.train_network(i,train_net=False)
            # self.nnet1.nnet.load_state_dict(self.nnet.nnet.state_dict())
            #self.nnet2.nnet.load_state_dict(self.nnet1.nnet.state_dict())
            #self.nnet3.nnet.load_state_dict(self.nnet1.nnet.state_dict())
            self.trainExamplesHistory.clear()

            # if i % 10 == 0:
            #     self.parallel_self_test_play(i)

            # self.trainExamplesHistory.append(iterationTrainExamples)
            # self.train_network(i)
            # self.trainExamplesHistory.clear()
            # self.parallel_self_test_play(i)
            
