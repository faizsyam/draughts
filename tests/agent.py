# %matplotlib inline
from numba import jit, cuda

import numpy as np
import numpy
import random

import torch
import torch.nn as nn

from torch.optim import Adam

from torch.utils.data import TensorDataset, DataLoader, Dataset

import MCTS as mc
from game import king_move_df, capture_move_df
from loss import softmax_cross_entropy_with_logits

import config
import loggers as lg
import time

import matplotlib.pyplot as plt
from IPython import display
import pylab as pl

from draughts1 import *
import pandas as pd

from timeit import default_timer as timer   


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class M(Dataset):
	def __init__(self,state,value,policy):
		super().__init__()
		self.state = state
		self.value = value
		self.policy = policy

	def __getitem__(self, i):
		return self.state[i], {'value':self.value[i], 'policy':self.policy[i]}

	def __len__(self):
		return len(self.state)

def get_normal_move_id(action,pos,move_df):
    move = print_move(action,pos)
    if move_is_capture(action,pos):
           pos_str = move.split('x')
    else:
        pos_str = move.split('-')

    from_ = int(pos_str[0])
    to_ = int(pos_str[1])

    # print(from_,to_)
    # display_position(pos)

    if not pos.is_white_to_move():
        from_ = 51 - from_
        to_ = 51 - to_

    try:
        move_id = move_df[(move_df['from']==from_)&(move_df['to']==to_)]['move_id'].iloc[0]
    except:
        print('')
        print('ERR ', from_, to_)
        print(pos.is_capture())
        print(move_is_capture(action,pos))
        display_position(pos)
        move_id = move_df[(move_df['from']==from_)&(move_df['to']==to_)]['move_id'].iloc[0]
    
    return move_id

def get_normal_move(move_id,pos,move_df):
    mv = move_df[move_df['move_id']==move_id]
    from_ = mv['from'].iloc[0]
    to_ = mv['to'].iloc[0]
    
    if not pos.is_white_to_move():
        from_ = 51 - from_
        to_ = 51 - to_
	
    if pos.is_capture():
        move_str = str(from_)+'x'+str(to_)
    else:
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

	
	def simulate(self, count):
		# lg.logger_mcts.info('ROOT NODE...%s', self.mcts.root.state.id)
		# self.mcts.root.state.render(lg.logger_mcts)
		# lg.logger_mcts.info('CURRENT PLAYER...%d', self.mcts.root.state.playerTurn)

		##### MOVE THE LEAF NODE
		leaf, value, done, breadcrumbs = self.mcts.moveToLeaf(count)
		# leaf.state.render(lg.logger_mcts)

		##### EVALUATE THE LEAF NODE
		value, breadcrumbs = self.evaluateLeaf(leaf, value, done, breadcrumbs)


		##### BACKFILL THE VALUE THROUGH THE TREE
		self.mcts.backFill(leaf, value, breadcrumbs)


	def act(self, state, tau, count):

		if self.mcts == None or state.id not in self.mcts.tree:
			self.buildMCTS(state)
		else:
			self.changeRootMCTS(state)


		#### run the simulation
		for sim in range(self.MCTSsimulations):
			# lg.logger_mcts.info('***************************')
			# lg.logger_mcts.info('****** SIMULATION %d ******', sim + 1)
			# lg.logger_mcts.info('***************************')
			self.simulate(count)

		#### get action values

		pi, values, move_ids = self.getAV(1,state.board.is_capture())

		####pick the action
		action, value = self.chooseAction(pi, values, tau, state)

		nextState, _, _, is_capture, _ = state.takeAction(action, count)
		
		# print('player: ',state.playerTurn,' move: ',print_move(action,state.board),' value: ',value)
		# display_position(nextState.board)
		# print(is_capture)
		NN_value = -self.get_preds(nextState, is_capture)[0]
		
		# lg.logger_mcts.info('ACTION VALUES...%s', pi)
		# lg.logger_mcts.info('CHOSEN ACTION...%d', action)
		# lg.logger_mcts.info('MCTS PERCEIVED VALUE...%f', value)
		# lg.logger_mcts.info('NN PERCEIVED VALUE...%f', NN_value)
		if state.board.is_capture():
			pi = pi[move_ids]

		return (action, pi, value, NN_value)


	def get_preds(self, state, is_capture):
		#predict the leaf
		

		allowedActions = state.allowedActions

		move_ids = []
		# print('get preds aa: ',len(allowedActions))
		# print('check')
		for move in allowedActions:
			# move_id = get_normal_move_id(move,state.board,normal_move_df)
			if is_capture:
				move_id = get_normal_move_id(move,state.board,capture_move_df)
			else:
				move_id = get_normal_move_id(move,state.board,king_move_df)
			move_ids.append(move_id)
		if is_capture:
			probs = list(np.zeros(2500,dtype=int))
			for idx, action in enumerate(allowedActions):
				newpos = state.board.succ(action)
				inputToModel = pos_to_numpy1(newpos)
				inputToModel = np.reshape(inputToModel, (4,10,10)) 
				inputToModel = torch.tensor(np.array([inputToModel],dtype=np.float64)).to(device)
				
				value = self.model(inputToModel.float(),is_capture)
				value = list(value.cpu().detach().numpy())[0][0]
				
				probs[move_ids[idx]] = value
			
			value = np.max(probs)
			probs = np.array(probs)
			# print(probs[move_ids])
			return ((value, probs, allowedActions, move_ids))
		else:
			inputToModel = self.model.convertToModelInput(state.binary)
			inputToModel = torch.tensor(np.array(inputToModel,dtype=np.float64)).to(device)

			# preds = self.model.predict(inputToModel)
			preds = self.model(inputToModel.float(),is_capture)
			value_array = preds[0].cpu().detach().numpy()
			logits_array = preds[1].cpu().detach().numpy()
			# print('vp: ',value_array,logits_array)
			value = value_array[0]

			logits = logits_array[0]
			mask = np.ones(logits.shape,dtype=bool)
			mask[move_ids] = False
			logits[mask] = -100

			#SOFTMAX
			odds = np.exp(logits)
			probs = odds / np.sum(odds) ###put this just before the for?

			return ((value, probs, allowedActions, move_ids))


	def evaluateLeaf(self, leaf, value, done, breadcrumbs):

		# lg.logger_mcts.info('------EVALUATING LEAF------')

		if done == 0:
			# poss = leaf.state.board
			# if leaf.state.playerTurn==-1:
			# 	poss = poss.flip()
			# display_position(poss)
			is_capture = leaf.state.board.is_capture()
			value, probs, allowedActions, move_ids = self.get_preds(leaf.state,is_capture)
			# print(type(probs))
			# print(is_capture)
			# print(len(probs))
			# probs = probs[move_ids]
			# print(len(probs))
			for idx, action in enumerate(allowedActions):
				newState, _, _, _, _ = leaf.state.takeAction(action)
				if newState.id not in self.mcts.tree:
					node = mc.Node(newState)
					self.mcts.addNode(node)
					# lg.logger_mcts.info('added node...%s...p = %f', node.id, probs[idx])
				else:
					node = self.mcts.tree[newState.id]
					# lg.logger_mcts.info('existing node...%s...', node.id)

				# if leaf.state.board.is_capture():
				# 	newEdge = mc.Edge(leaf, node, 1/len(allowedActions), action)
				# 	move_id = -1
				# else:
				# print('newedge')
				newEdge = mc.Edge(leaf, node, probs[move_ids[idx]], action)
				# move_id = get_move_id(action,leaf.state.board,move_df)
				# move_id = get_normal_move_id(action,leaf.state.board,normal_move_df)
				# if is_capture:
				# 	move_id = get_normal_move_id(action,leaf.state.board,capture_move_df)
				# else:
				# 	try:
				# 		move_id = get_normal_move_id(action,leaf.state.board,king_move_df)
				# 	except:
				# 		display_position(leaf.state.board)
				# 		print('move:',print_move(action,leaf.state.board))
				# 		print('pos:',leaf.state.board)
				# 		move_id = get_normal_move_id(action,leaf.state.board,king_move_df)
				# print('move_id',move_id)
				# print('done')
				leaf.edges.append((action, move_ids[idx], newEdge))
				
		# else:
		# 	lg.logger_mcts.info('GAME VALUE FOR %d: %f', leaf.playerTurn, value)

		return ((value, breadcrumbs))

	def getAV(self, tau,is_capture):
		edges = self.mcts.root.edges

		if is_capture:
			action_size = 2500
		else:
			action_size = 570

		pi = np.zeros(action_size, dtype=np.integer)
		values = np.zeros(action_size, dtype=np.float32)
		
		move_ids = []
		for action, move_id, edge in edges:
			pi[move_id] = pow(edge.stats['N'], 1/tau)
			values[move_id] = edge.stats['Q']
			move_ids.append(move_id)

		move_ids = np.array(move_ids)

		if np.sum(pi)!=0:
			pi = pi / (np.sum(pi) * 1.0)
		else:
			pi = pi / (0.00001 * 1.0)
			
		if len(pi)==0:
			print('idk')
		return pi, values, move_ids


	def chooseAction(self, pi, values, tau, state):
		if tau == 0:
			actions = np.argwhere(pi == max(pi))
			try:
				action = random.choice(actions)[0]
			except:
				print('ERR | actions: ',actions)
				print(pi)
				action = actions[0]
		else:
			action_idx = np.random.multinomial(1, pi)
			action = np.where(action_idx==1)[0][0]

		value = values[action]

		# print('av: ',action, value)
		# action_move = get_normal_move(action,state.board,normal_move_df)
		if state.board.is_capture():
			action_move = get_normal_move(action,state.board,capture_move_df)
		else:
			action_move = get_normal_move(action,state.board,king_move_df)

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

	def convertIDToPos(self, board_id):
		text = '.'*50
		idd = np.array([*board_id])
		idx = np.where(np.array(idd)=='1')[0]
		# print(idx)
		chrs = ['o','x','O','X']
		for i in idx:
			index = i%50
			text = text[:index] + chrs[int(i/50)] + text[index + 1:]
		text += 'W'
		pos = parse_position(text)
		return pos

	def convertIDToModelInput(self, board_id):
		# print('.',end='')
		idn = np.array([*board_id])
		idn = idn.astype(int)
		idn = np.insert(idn,ids,0).reshape(4,10,10)
		return np.array(idn)

	def convertPosToModelInput(self,pos):
		ppos = pos
		if not ppos.is_white_to_move():
			ppos.flip()
		result = np.array(pos_to_numpy1(ppos)).reshape(4,10,10)
		result = result.astype(int)
		return result

	def convertPosToId(self,pos):
		
		board = np.array(np.zeros(200),dtype=int)
		ppos = pos
		if not ppos.is_white_to_move():
			ppos.flip()
		text = print_position(ppos, False, True)
		text = np.array([*text[:-1]])

		for i,c in enumerate(['o','x','O','X']):
			idx = np.where(np.array(text)==c)[0]
			board[idx+50*i]=1
		result = ''.join(map(str,board))
		return result

	def replay(self, ltmemory):
		lg.logger_mcts.info('******RETRAINING MODEL******')

		losses = []

		optimizer = Adam(self.model.parameters(), lr=0.001, weight_decay=0.0001)
		mse1 = nn.L1Loss()
		# mse2 = nn.MSELoss()
		cet1 = nn.L1Loss()
		cet2 = nn.L1Loss()
		
		self.model.to(device)

		# loss_val_cap = []
		loss_val_ncap = []
		loss_pol_cap = []
		loss_pol_ncap = []

		training_states_ncap = []
		training_states_cap = []
		training_value_ncap = []
		training_value_cap = []
		training_policy_ncap = []
		training_policy_cap = []
		count = 0
		group_ids = []
		group_borders = []
		for row in ltmemory:
			value = row['value']
			policy = row['AV']
			if row['is_capture']:
				init_state = self.convertIDToPos(row['id'])
				moves = generate_moves(init_state)
				for move in moves:
					state = init_state.succ(move)
					state = self.convertPosToModelInput(state)
					training_states_cap.append(state)
					training_value_cap.append(value)
					group_ids.append(len(group_borders))
					count += 1
				group_borders.append(count)
				for i in policy:
					training_policy_cap.append(i)
			else:
				state = self.convertIDToModelInput(row['id'])
				training_states_ncap.append(state)
				training_value_ncap.append(value)
				training_policy_ncap.append(policy)

		training_states_ncap = np.array(training_states_ncap,dtype=np.float64)
		training_states_cap = np.array(training_states_cap,dtype=np.float)
		
		training_value_ncap = np.array(training_value_ncap,dtype=np.float64)
		training_value_cap = np.array(training_value_cap,dtype=np.float64)
		
		training_policy_ncap = np.array(training_policy_ncap,dtype=np.float64)
		training_policy_cap = np.array(training_policy_cap,dtype=np.float64)

		training_states_ncap = torch.tensor(training_states_ncap,dtype=torch.float)
		training_states_cap = torch.tensor(training_states_cap,dtype=torch.float)

		training_value_ncap = torch.tensor(training_value_ncap,dtype=torch.float,requires_grad=True)
		training_value_ncap = training_value_ncap[:,None]
		training_value_cap = torch.tensor(training_value_cap,dtype=torch.float,requires_grad=True)
		training_value_cap = training_value_cap[:,None]

		training_policy_ncap = torch.tensor(training_policy_ncap,dtype=torch.float,requires_grad=True)
		training_policy_cap = torch.tensor(training_policy_cap,dtype=torch.float,requires_grad=True)
		training_policy_cap = training_policy_cap[:,None]
		
		training_states_ncap = training_states_ncap.to(device)
		training_states_cap = training_states_cap.to(device)
		training_value_ncap = training_value_ncap.to(device)
		training_value_cap = training_value_cap.to(device)
		training_policy_ncap = training_policy_ncap.to(device)
		training_policy_cap = training_policy_cap.to(device)

		dataset_ncap = M(training_states_ncap,training_value_ncap,training_policy_ncap)
		dataset_cap = M(training_states_cap,training_value_cap,training_policy_cap)
		dataloader_ncap = DataLoader(dataset_ncap, batch_size=config.BATCH_SIZE)
		dataloader_cap = DataLoader(dataset_cap, batch_size=config.BATCH_SIZE)

		self.model.train(True)
		print('len ncap:',len(training_states_ncap),'len cap:',len(training_states_cap))

		for i in range(config.TRAINING_LOOPS):
			print(i,end=' ')
			# minibatch = random.sample(ltmemory, min(config.BATCH_SIZE, len(ltmemory)))
			for x, y in dataloader_ncap:
				
				optimizer.zero_grad()
				pred_value_ncap, pred_policy_ncap = self.model(x,False)
				
				loss_v_ncap = mse1(pred_value_ncap, y['value'])
				loss_p_ncap = cet1(pred_policy_ncap, y['policy'])
				# print(pred_policy_ncap)
				# print(y['policy'])
				# print(loss_p_ncap)
				loss = loss_v_ncap + loss_p_ncap
				loss.backward()
				optimizer.step()

				loss_val_ncap.append(loss_v_ncap.cpu().detach().numpy())
				loss_pol_ncap.append(loss_p_ncap.cpu().detach().numpy())

			for x, y in dataloader_cap:

				optimizer.zero_grad()
				pred_policy_cap = self.model(x,True)

				# count = 0
				# tpc = pred_policy_cap.cpu().detach().numpy()
				# pred_value_cap_max = np.zeros(len(tpc))
				# for i in group_borders:
				# 	for j in range(count,i):
				# 		pred_value_cap_max[j] = np.max(tpc[count:i])
				# 	count = i
				# pred_value_cap_max = pred_value_cap_max[:,None]
				# pred_value_cap_max = torch.tensor(pred_value_cap_max,dtype=torch.float,requires_grad=True)
				# pred_value_cap_max = pred_value_cap_max.to(device)
				
				# loss_v_cap = mse2(pred_value_cap_max, y['value'])
				# loss_p_cap = cet2(pred_policy_cap, y['policy'])
				loss_p_cap = cet2(pred_policy_cap, y['value'])
		
				# print(pred_policy_cap)
				# print(y['value'])
				# print(loss_p_cap)
				loss = loss_p_cap # + loss_v_cap
				loss.backward()
				optimizer.step()

				# loss_val_cap.append(loss_v_cap.cpu().detach().numpy())
				loss_pol_cap.append(loss_p_cap.cpu().detach().numpy())
				# losses.append(loss.item())

		print('')
		# print(losses)

		# loss_val_cap = np.mean(np.array(loss_val_cap))
		loss_val_ncap = np.mean(np.array(loss_val_ncap))
		loss_pol_ncap = np.mean(np.array(loss_pol_ncap))
		loss_pol_cap = np.mean(np.array(loss_pol_cap))
		print([loss_val_ncap,loss_pol_ncap,loss_pol_cap])
		# plt.plot(self.train_overall_loss, 'k')
		# plt.plot(self.train_value_loss, 'k:')
		# plt.plot(self.train_policy_loss, 'k--')

		# plt.legend(['train_overall_loss', 'train_value_loss', 'train_policy_loss'], loc='lower left')

		# display.clear_output(wait=True)
		# display.display(pl.gcf())
		# pl.gcf().clear()
		# time.sleep(1.0)

		# print('\n')
		# self.model.printWeightAverages()

	def predict(self, inputToModel):
		preds = self.model.predict(inputToModel)
		return preds

	def buildMCTS(self, state):
		# lg.logger_mcts.info('****** BUILDING NEW MCTS TREE FOR AGENT %s ******', self.name)
		# print('build ', state.id)
		self.root = mc.Node(state)
		self.mcts = mc.MCTS(self.root, self.cpuct)

	def changeRootMCTS(self, state):
		# print('change ', state.id)
		# lg.logger_mcts.info('****** CHANGING ROOT OF MCTS TREE TO %s FOR AGENT %s ******', state.id, self.name)
		self.mcts.root = self.mcts.tree[state.id]

ids = []
for i in range(201):
    if i%10!=5:
        ids.append(i)
        if (i%10==0) & (i!=0) & (i!=200):
            ids.append(i)