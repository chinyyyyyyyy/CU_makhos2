import math
import numpy as np
import random
EPS = 1e-8


class MCTS():
    """
    This class handles the MCTS tree.
    """

    def __init__(self, game, nnet, args, eval=False, verbose=False):
        self.game = game 
        self.nnet = nnet
        self.args = args
        self.verbose = verbose
        self.eval = eval    # eval mode
        
        
        self.states_visited = 0
        self.Qsa = {}       # stores Q values for s,a (as defined in the paper)
        self.Nsa = {}       # stores #times edge s,a was visited
        self.Ns = {}        # stores #times board s was visited
        
        self.Ps = {}      # stores initial policy (returned by neural net)
        self.Es = {}    # stores game.getGameEnded ended for board s
        self.Vs = {}        # stores game.getValidMoves for board s

    def getActionProb(self, boardHistory, temp=1):
        """
        This function performs numMCTSSims simulations of MCTS starting from
        canonicalBoard.

        Returns:
            probs: a policy vector where the probability of the ith action is
                   proportional to Nsa[(s,a)]**(1./temp)
        """
        # print('len getprob:', len(boardHistory))
        cur_move = self.game.gameState.turn
        cur_stale = self.game.gameState.stale
        self.state_visits = 0
        for i in range(self.args.numMCTSSims):
            self.search(boardHistory, is_search_root=True)
            self.game.gameState.turn = cur_move
            self.game.gameState.stale = cur_stale

        s = self.game.stringRepresentation(boardHistory)
        # print('string rep:',s)
        counts = [self.Nsa[(s, a)] if (
            s, a) in self.Nsa else 0 for a in range(self.game.getActionSize())]
        
        

        if temp == 0:
            bestA = np.argmax(counts)
            probs = [0]*len(counts)
            probs[bestA] = 1
            if self.verbose:
                try:
                    print('MCTS States visited:',self.states_visited)
                    print('Win confidence for current board:' +  str( round( (0.5 + self.Qsa[(s,bestA)][0]/2)*100,2) ) + '%')
                except:
                    pass
                #print((self.game.gameState.turn, self.game.gameState.stale))
            return probs

        counts = [x**(1./temp) for x in counts]
        probs = [x/float(sum(counts)) for x in counts]
        return probs

    def search(self, boardHistory, is_search_root=False):
        """
        This function performs one iteration of MCTS. It is recursively called
        till a leaf node is found. The action chosen at each node is one that
        has the maximum upper confidence bound as in the paper.

        Once a leaf node is found, the neural network is called to return an
        initial policy P and a value v for the state. This value is propogated
        up the search path. In case the leaf node is a terminal state, the
        outcome is propogated up the search path. The values of Ns, Nsa, Qsa are
        updated.

        NOTE: the return values are the negative of the value of the current
        state. This is done since v is in [-1,1] and if v is the value of a
        state for the current player, then its value is -v for the other player.

        Returns:
            v: the negative of the value of the current canonicalBoard
        """
        self.states_visited += 1
        s = self.game.stringRepresentation(boardHistory)
        canonicalBoard = boardHistory
    

        if s not in self.Es:
            self.Es[s] = self.game.getGameEnded(canonicalBoard, 1)

        if self.Es[s] != 0:
            return -self.Es[s]

        if s not in self.Ps:
            # leaf node
            valids = self.game.getValidMoves(canonicalBoard, 1)
            self.Ps[s], v = self.nnet.predict(
                boardHistory, self.game.gameState.turn, self.game.gameState.stale, valids)
            

            # self.Ps[s] = self.Ps[s]*valids      # masking invalid moves
            sum_Ps_s = np.sum(self.Ps[s])
            # print(sum_Ps_s)

            # if sum_Ps_s > 0:
            #     self.Ps[s] /= sum_Ps_s    # renormalize
            if sum_Ps_s <= 0:
                # if all valid moves were masked make all valid moves equally probable

                # NB! All valid moves may be masked if either your NNet architecture is insufficient or you've get overfitting or something else.
                # If you have got dozens or hundreds of these messages you should pay attention to your NNet and/or training process.
                print("All valid moves were masked, do workaround.")
                self.Ps[s] = self.Ps[s] + valids
                self.Ps[s] /= np.sum(self.Ps[s])
	
            self.Vs[s] = valids
            self.Ns[s] = 0
            # print('return not in ps')
            return -v

        if is_search_root and not self.eval:
            dir_noise = np.random.dirichlet([1]*32*32)
            tempps = 0.75*self.Ps[s] + 0.25*dir_noise
        else:
            tempps = self.Ps[s]

        valids = self.Vs[s]
        cur_best = -float('inf')
        best_act = []

        # pick the action with the highest upper confidence bound
        for a in range(self.game.getActionSize()):
            if valids[a]:
                if (s, a) in self.Qsa:
                    u = self.Qsa[(s, a)] + self.args.cpuct*tempps[a] * \
                        math.sqrt(self.Ns[s])/(1+self.Nsa[(s, a)])
                else:
                    u = self.args.cpuct * \
                        tempps[a]*math.sqrt(self.Ns[s] + EPS)     # Q = 0 ?

                if u > cur_best:
                    cur_best = u
                    best_act = [a]
                elif u == cur_best:
                    best_act.append(a)

        a = random.choice(best_act)

        # a = best_act
        next_s, next_player = self.game.getNextState(canonicalBoard, 1, a)
        next_s = self.game.getCanonicalForm(next_s, next_player)
        # newHistory = boardHistory.copy()
        # newHistory.append(next_s)

        v = self.search(next_s)

        if (s, a) in self.Qsa:
            self.Qsa[(s, a)] = (self.Nsa[(s, a)] *
                                self.Qsa[(s, a)] + v)/(self.Nsa[(s, a)]+1)
            self.Nsa[(s, a)] += 1

        else:
            self.Qsa[(s, a)] = v
            self.Nsa[(s, a)] = 1

        self.Ns[s] += 1
        return -v

    # def reset_tree(self):
    #     self.Qsa = {}       # stores Q values for s,a (as defined in the paper)
    #     self.Nsa = {}       # stores #times edge s,a was visited
    #     self.Ns = {}        # stores #times board s was visited
    #     self.Ps = {}        # stores initial policy (returned by neural net)
    #     self.Es = {}        # stores game.getGameEnded ended for board s
    #     self.Vs = {}        # stores game.getValidMoves for board s
