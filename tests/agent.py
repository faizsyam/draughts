# %matplotlib inline

import numpy as np
import random

import MCTS as mc
from game import normal_move_df
from loss import softmax_cross_entropy_with_logits

import config
import loggers as lg
import time

import matplotlib.pyplot as plt
from IPython import display
import pylab as pl

from draughts1 import *
import pandas as pd

def get_normal_move_id(action,pos,move_df):
    move = print_move(action,pos)
    if move_is_capture(action,pos):
           pos_str = move.split('x')
    else:
        pos_str = move.split('-')

    from_ = int(pos_str[0])
    to_ = int(pos_str[1])

    print(from_,to_)
    move_id = move_df[(move_df['from']==from_)&(move_df['to']==to_)]['move_id'].iloc[0]
    
    return move_id

def get_normal_move(move_id,pos,move_df):
    mv = move_df[move_df['move_id']==move_id]
    from_ = mv['from'].iloc[0]
    to_ = mv['to'].iloc[0]
    
    move_str = str(from_)+'-'+str(to_)
    
    return parse_move(move_str,pos)

# def get_move_id(action, pos, move_df):
# 	move = print_move(action,pos)
# 	if move_is_capture(action,pos):
# 		pos_str = move.split('x')
# 	else:
# 		pos_str = move.split('-')

# 	from_ = int(pos_str[0])
# 	to_ = int(pos_str[1])

# 	id = move_df[(move_df['start']==from_)&(move_df['end']==to_)].iloc[0]['index']
# 	return id

# def get_move_ids(allowedActions, pos, move_df):

# 	move_ids = []

# 	for action in allowedActions:
# 		# print(print_position(pos, False, True))
# 		move = print_move(action,pos)
# 		if move_is_capture(action,pos):
# 			pos_str = move.split('x')
# 		else:
# 			pos_str = move.split('-')

# 		from_ = int(pos_str[0])
# 		to_ = int(pos_str[1])

# 		# print(from_,to_)
# 		try:
# 			id = move_df[(move_df['start']==from_)&(move_df['end']==to_)].iloc[0]['index']
# 			move_ids.append(id)
# 		except:
# 			aa = 1
			
# 	return move_ids

class User():
	def __init__(self, name, state_size, action_size):
		self.name = name
		self.state_size = state_size
		self.action_size = action_size

	def act(self, state, tau):
		action = input('Enter your chosen action: ')
		pi = np.zeros(self.action_size)
		pi[action] = 1
		value = None
		NN_value = None
		return (action, pi, value, NN_value)



class Agent():
	def __init__(self, name, state_size, action_size, mcts_simulations, cpuct, model):
		self.name = name

		self.state_size = state_size
		self.action_size = action_size

		self.cpuct = cpuct

		self.MCTSsimulations = mcts_simulations
		self.model = model

		self.mcts = None

		self.train_overall_loss = []
		self.train_value_loss = []
		self.train_policy_loss = []
		self.val_overall_loss = []
		self.val_value_loss = []
		self.val_policy_loss = []

	
	def simulate(self):
		print('simulate')
		lg.logger_mcts.info('ROOT NODE...%s', self.mcts.root.state.id)
		self.mcts.root.state.render(lg.logger_mcts)
		lg.logger_mcts.info('CURRENT PLAYER...%d', self.mcts.root.state.playerTurn)

		##### MOVE THE LEAF NODE
		leaf, value, done, breadcrumbs = self.mcts.moveToLeaf()
		leaf.state.render(lg.logger_mcts)

		##### EVALUATE THE LEAF NODE
		value, breadcrumbs = self.evaluateLeaf(leaf, value, done, breadcrumbs)

		##### BACKFILL THE VALUE THROUGH THE TREE
		self.mcts.backFill(leaf, value, breadcrumbs)


	def act(self, state, tau):

		if self.mcts == None or state.id not in self.mcts.tree:
			self.buildMCTS(state)
		else:
			self.changeRootMCTS(state)

		#### run the simulation
		for sim in range(self.MCTSsimulations):
			lg.logger_mcts.info('***************************')
			lg.logger_mcts.info('****** SIMULATION %d ******', sim + 1)
			lg.logger_mcts.info('***************************')
			self.simulate()

		#### get action values
		pi, values = self.getAV(1)

		####pick the action
		action, value = self.chooseAction(pi, values, tau, state)

		nextState, _, _ = state.takeAction(action)

		NN_value = -self.get_preds(nextState)[0]

		lg.logger_mcts.info('ACTION VALUES...%s', pi)
		lg.logger_mcts.info('CHOSEN ACTION...%d', action)
		lg.logger_mcts.info('MCTS PERCEIVED VALUE...%f', value)
		lg.logger_mcts.info('NN PERCEIVED VALUE...%f', NN_value)

		return (action, pi, value, NN_value)


	def get_preds(self, state):
		#predict the leaf
		inputToModel = self.model.convertToModelInput(state)

		# preds = self.model.predict(inputToModel)
		preds = self.model(inputToModel.float())
		value_array = preds[0]
		logits_array = preds[1]
		value = value_array[0]

		logits = logits_array[0]

		allowedActions = state.allowedActions

		move_ids = []

		for move in allowedActions:
			move_id = get_normal_move_id(move,state.board,normal_move_df)
			move_ids.append(move_id)

		mask = np.ones(logits.shape,dtype=bool)
		mask[move_ids] = False
		logits[mask] = -100

		#SOFTMAX
		odds = np.exp(logits.detach().numpy())
		probs = odds / np.sum(odds) ###put this just before the for?

		return ((value, probs, allowedActions, move_ids))


	def evaluateLeaf(self, leaf, value, done, breadcrumbs):

		lg.logger_mcts.info('------EVALUATING LEAF------')

		if done == 0:
			poss = leaf.state.board
			if leaf.state.playerTurn==-1:
				poss = poss.flip()
			display_position(poss)
			value, probs, allowedActions, move_ids = self.get_preds(leaf.state)
			lg.logger_mcts.info('PREDICTED VALUE FOR %d: %f', leaf.state.playerTurn, value)

			probs = probs[move_ids]

			for idx, action in enumerate(allowedActions):
				newState, _, _ = leaf.state.takeAction(action)
				if newState.id not in self.mcts.tree:
					node = mc.Node(newState)
					# print('add ',node.id)
					self.mcts.addNode(node)
					# lg.logger_mcts.info('added node...%s...p = %f', node.id, probs[idx])
				else:
					node = self.mcts.tree[newState.id]
					lg.logger_mcts.info('existing node...%s...', node.id)

				if leaf.state.board.is_capture():
					newEdge = mc.Edge(leaf, node, 1/len(allowedActions), action)
					move_id = -1
				else:
					newEdge = mc.Edge(leaf, node, probs[idx], action)
					# move_id = get_move_id(action,leaf.state.board,move_df)
					move_id = get_normal_move_id(action,leaf.state.board,normal_move_df)
				# print('move_id',move_id)

				leaf.edges.append((action, move_id, newEdge))
				
		else:
			lg.logger_mcts.info('GAME VALUE FOR %d: %f', leaf.playerTurn, value)

		return ((value, breadcrumbs))


		
	def getAV(self, tau):
		edges = self.mcts.root.edges
		pi = np.zeros(self.action_size, dtype=np.integer)
		values = np.zeros(self.action_size, dtype=np.float32)
		
		for action, move_id, edge in edges:
			# print(move_id)
			pi[move_id] = pow(edge.stats['N'], 1/tau)
			values[move_id] = edge.stats['Q']

		pi = pi / (np.sum(pi) * 1.0)
		return pi, values

	def chooseAction(self, pi, values, tau, state):
		if tau == 0:
			actions = np.argwhere(pi == max(pi))
			action = random.choice(actions)[0]
		else:
			action_idx = np.random.multinomial(1, pi)
			action = np.where(action_idx==1)[0][0]

		value = values[action]

		action_move = get_normal_move(action,state.board,normal_move_df)
		# mdf = move_df[move_df['index']==action]
		# xx = 'x' if state.board.is_capture() else '-'
		# start = mdf['start'].iloc[0]
		# endd = mdf['end'].iloc[0]
		
		# str_move = str(mdf['start'].iloc[0]) + xx + str(mdf['end'].iloc[0])
		# print('HALO ', str_move)
		# action_move = parse_move(str_move,state.board)
		# print('MOVE ', print_move(action_move,state.board))
		# print(pos_to_numpy1(state.board).reshape(4,10,10))

		return action_move, value

	def replay(self, ltmemory):
		lg.logger_mcts.info('******RETRAINING MODEL******')


		for i in range(config.TRAINING_LOOPS):
			minibatch = random.sample(ltmemory, min(config.BATCH_SIZE, len(ltmemory)))

			training_states = np.array([self.model.convertToModelInput(row['state']) for row in minibatch])
			training_targets = {'value_head': np.array([row['value'] for row in minibatch])
								, 'policy_head': np.array([row['AV'] for row in minibatch])} 

			fit = self.model.fit(training_states, training_targets, epochs=config.EPOCHS, verbose=1, validation_split=0, batch_size = 32)
			lg.logger_mcts.info('NEW LOSS %s', fit.history)

			self.train_overall_loss.append(round(fit.history['loss'][config.EPOCHS - 1],4))
			self.train_value_loss.append(round(fit.history['value_head_loss'][config.EPOCHS - 1],4)) 
			self.train_policy_loss.append(round(fit.history['policy_head_loss'][config.EPOCHS - 1],4)) 

		plt.plot(self.train_overall_loss, 'k')
		plt.plot(self.train_value_loss, 'k:')
		plt.plot(self.train_policy_loss, 'k--')

		plt.legend(['train_overall_loss', 'train_value_loss', 'train_policy_loss'], loc='lower left')

		display.clear_output(wait=True)
		display.display(pl.gcf())
		pl.gcf().clear()
		time.sleep(1.0)

		print('\n')
		self.model.printWeightAverages()

	def predict(self, inputToModel):
		preds = self.model.predict(inputToModel)
		return preds

	def buildMCTS(self, state):
		lg.logger_mcts.info('****** BUILDING NEW MCTS TREE FOR AGENT %s ******', self.name)
		print('build ', state.id)
		self.root = mc.Node(state)
		self.mcts = mc.MCTS(self.root, self.cpuct)

	def changeRootMCTS(self, state):
		print('change ', state.id)
		lg.logger_mcts.info('****** CHANGING ROOT OF MCTS TREE TO %s FOR AGENT %s ******', state.id, self.name)
		self.mcts.root = self.mcts.tree[state.id]