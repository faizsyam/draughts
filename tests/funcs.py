import numpy as np
import random

import loggers as lg
from draughts1 import *

from game import Game, GameState
from model import Residual_CNN
from model2 import CNN_Net

import torch
from agent import Agent, User
from settings import run_folder, run_archive_folder
import config

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def playMatchesBetweenVersions(env, player1_NN, player2_NN, EPISODES, turns_until_tau0, goes_first = 0):
    
    logger = lg.logger_tourney
    # if player1version == -1:
    #     player1 = User('player1', env.state_size, env.action_size)
    # else:
    #     # player1_NN = Residual_CNN(config.REG_CONST, config.LEARNING_RATE, env.input_shape,   env.action_size, config.HIDDEN_CNN_LAYERS)
    #     # player1_NN = CNN_Net().to(device)


    #     # if player1version > 0:
    #     #     # player1_network = player1_NN.read(env.name, run_version, player1version)
    #     #     # player1_NN.model.set_weights(player1_network.get_weights())  
    #     #     player1_NN.load_state_dict(torch.load(run_folder + 'models2/weights_v' + "{0:0>4}".format(player1version)))
    #     player1_NN = torch.load(run_folder + 'models3/model_v' + "{0:0>4}".format(player1version))
    #     # player1_NN.load_state_dict(torch.load(run_folder + 'models3/weights_v' + "{0:0>4}".format(player1version)))
    #     player1 = Agent('player1', env.state_size, env.action_size, config.MCTS_SIMS, config.CPUCT, player1_NN)
    player1 = Agent('player1', env.state_size, env.action_size, config.MCTS_SIMS, config.CPUCT, player1_NN)

    # if player2version == -1:
    #     player2 = User('player2', env.state_size, env.action_size)
    # else:
    #     # player2_NN = Residual_CNN(config.REG_CONST, config.LEARNING_RATE, env.input_shape,   env.action_size, config.HIDDEN_CNN_LAYERS)
    #     # player2_NN = CNN_Net().to(device)
        
    #     # if player2version > 0:
    #     #     # player2_network = player2_NN.read(env.name, run_version, player2version)
    #     #     # player2_NN.model.set_weights(player2_network.get_weights())
    #     #     player2_NN.load_state_dict(torch.load(run_folder + 'models2/weights_v' + "{0:0>4}".format(player1version)))
    #     player2_NN = torch.load(run_folder + 'models2/model_v' + "{0:0>4}".format(player2version))
    #     # player2_NN.load_state_dict(torch.load(run_folder + 'models2/weights_v' + "{0:0>4}".format(player2version)))
    #     player2 = Agent('player2', env.state_size, env.action_size, config.MCTS_SIMS, config.CPUCT, player2_NN)
    player2 = Agent('player2', env.state_size, env.action_size, config.MCTS_SIMS, config.CPUCT, player2_NN)

    scores, memory, points, sp_scores = playMatches(player1, player2, EPISODES, logger, turns_until_tau0, None, goes_first)

    return (scores, memory, points, sp_scores)


def playMatches(player1, player2, EPISODES, logger, turns_until_tau0, memory = None, goes_first = 0):

    env = Game()
    scores = {player1.name:0, "drawn": 0, player2.name:0}
    sp_scores = {'sp':0, "drawn": 0, 'nsp':0}
    points = {player1.name:[], player2.name:[]}
    
    for e in range(EPISODES):
        # print('Episode: ', e+1)
        # logger.info('====================')
        # logger.info('EPISODE %d OF %d', e+1, EPISODES)
        # logger.info('====================')

        print (str(e+1) + ' ', end='')

        state = env.reset()
        
        done = 0
        turn = 0
        player1.mcts = None
        player2.mcts = None

        if goes_first == 0:
            player1Starts = random.randint(0,1) * 2 - 1
        else:
            player1Starts = goes_first

        if player1Starts == 1:
            players = {1:{"agent": player1, "name":player1.name}
                    , -1: {"agent": player2, "name":player2.name}
                    }
            logger.info(player1.name + ' plays as X')
        else:
            players = {1:{"agent": player2, "name":player2.name}
                    , -1: {"agent": player1, "name":player1.name}
                    }
            logger.info(player2.name + ' plays as X')
            logger.info('--------------')

        # env.gameState.render(logger)

        count = 0
        # print('XXX',end=' ')
        while done == 0:
            # try:
            #     tboard = state.board
            #     if state.playerTurn==-1:
            #         tboard.flip()
            #     display_position(tboard)
            # except:
            #     print(state.board)
            # print('.',end='')
            # print('turn: ', turn,' player: ',state.playerTurn)
            turn = turn + 1
    
            #### Run the MCTS algo and return an action
            if turn < turns_until_tau0:
                action, pi, MCTS_value, NN_value = players[state.playerTurn]['agent'].act(state, 1, count)
            else:
                action, pi, MCTS_value, NN_value = players[state.playerTurn]['agent'].act(state, 0, count)

            if memory != None:
                ####Commit the move to memory
                # if state.board.is_capture():
                    # ones=np.where(pi == 1)[0]
                    # print(pi.shape,ones)
                    # if len(ones)==0:
                    # print(state.board)
                memory.commit_stmemory(env.identities, state, pi)

            # print(state.board)
            # print('MCTS perceived value for %s: %f', state.playerTurn ,np.round(MCTS_value,2))
            # print('NN perceived value for %s: %f', state.playerTurn, np.round(NN_value,2))
            # print(print_move(action,state.board))
            # logger.info('action: %d', action)
            # for r in range(env.grid_shape[0]):
            #     logger.info(['----' if x == 0 else '{0:.2f}'.format(np.round(x,2)) for x in pi[env.grid_shape[1]*r : (env.grid_shape[1]*r + env.grid_shape[1])]])
            # logger.info('MCTS perceived value for %s: %f', state.pieces[str(state.playerTurn)] ,np.round(MCTS_value,2))
            # logger.info('NN perceived value for %s: %f', state.pieces[str(state.playerTurn)] ,np.round(NN_value,2))
            # logger.info('====================')

            ### Do the action
            # print('S',count,end=': ')
            state, value, done, _, count = env.step(action, count) #the value of the newState from the POV of the new playerTurn i.e. -1 if the previous player played a winning move
            # print(end=' | ')
            # env.gameState.render(logger)
            
            # CP LOSE
            # value 1: black wins -> CP=W, -1: white wins -> CP=B

            # if memory==None:
            #     print(print_move(action))
            #     display_position(state.board)
            # print(state.playerTurn)
            # display_position(state.board)
            if done == 1: 
                # print('cp ',state.playerTurn,'val ',value)
                # print(state.board)
                # print('res: ',state.board.result(state.board.turn()),' val: ',value)
                # print('DONE. Winner: ',state.playerTurn if value==1 else -state.playerTurn)
                # print('turn ',state.playerTurn,'val ',value,'res ',state.board.result(state.board.turn()),'board turn ',state.board.turn())
                if memory != None:
                    #### If the game is finished, assign the values correctly to the game moves
                    for move in memory.stmemory:
                        if value==0:
                            move['value'] = 0
                        elif move['playerTurn'] == value: #state.playerTurn:
                            move['value'] = -1 #value
                        else:
                            move['value'] = 1 #-value
                         
                    memory.commit_ltmemory()
             
                # if value == 1:
                #     logger.info('%s WINS!', players[state.playerTurn]['name'])
                #     scores[players[state.playerTurn]['name']] = scores[players[state.playerTurn]['name']] + 1
                #     if state.playerTurn == 1: 
                #         sp_scores['sp'] = sp_scores['sp'] + 1
                #     else:
                #         sp_scores['nsp'] = sp_scores['nsp'] + 1

                # elif value == -1:
                # logger.info('%s WINS!', players[-state.playerTurn]['name'])
                if value!=0:
                    scores[players[-value]['name']] = scores[players[-value]['name']] + 1
                
                    if value == 1: 
                        sp_scores['nsp'] = sp_scores['nsp'] + 1
                    else:
                        sp_scores['sp'] = sp_scores['sp'] + 1
                    points[players[value]['name']].append(-1)
                    points[players[-value]['name']].append(1)

                else:
                    # logger.info('DRAW...')
                    scores['drawn'] = scores['drawn'] + 1
                    sp_scores['drawn'] = sp_scores['drawn'] + 1
                    points[players[state.playerTurn]['name']].append(0)
                    points[players[-state.playerTurn]['name']].append(0)

        # print('')

    return (scores, memory, points, sp_scores)
