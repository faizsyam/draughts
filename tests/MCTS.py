import numpy as np
import logging
import config
from numba import jit, cuda

from draughts1 import *
from utils import setup_logger
import loggers as lg

class Node():

	def __init__(self, state, count = 0):
		self.state = state
		self.playerTurn = state.playerTurn
		self.id = state.id
		self.edges = []
		self.count = count

	def isLeaf(self):
		if len(self.edges) > 0:
			return False
		else:
			return True

class Edge():

	def __init__(self, inNode, outNode, prior, action):
		self.id = inNode.state.id + '|' + outNode.state.id
		self.inNode = inNode
		self.outNode = outNode
		self.playerTurn = inNode.state.playerTurn
		self.action = action

		self.stats =  {
					'N': 0,
					'W': 0,
					'Q': 0,
					'P': prior,
				}
		
class MCTS():

	def __init__(self, root, cpuct):
		self.root = root
		self.tree = {}
		self.cpuct = cpuct
		self.addNode(root)
	
	def __len__(self):
		return len(self.tree)

	def moveToLeaf(self,count):
		
		# lg.logger_mcts.info('------MOVING TO LEAF------')

		breadcrumbs = []
		currentNode = self.root

		done = 0
		value = 0
		cc = 0
		while (not currentNode.isLeaf()) & (done==0):
			# print('1 ',end='')
			if cc >= 99999:
				print('break',end=' ')
				# display_position(currentNode.state.board)
				break
			# 	# cek = currentNode.state.board.white_man_count()==0 & currentNode.state.board.black_man_count()==0 & currentNode.state.board.white_king_count()>0 & currentNode.state.board.black_king_count()>0
				# print('Noonono')
			# 	print(currentNode.state.board.white_man_count(),end=' ')
			# 	print(currentNode.state.board.black_man_count(),end=' ')
			# 	print(currentNode.state.board.white_king_count(),end=' ')
			# 	print(currentNode.state.board.black_king_count(),end=' ')
			# 	# # display_position(currentNode.state.board)
				# print(currentNode.state.board)
			# 	# # print(currentNode.state.board.is_end())

				# Scan.set("variant", "normal")
				# Scan.set("book", "false")
				# Scan.set("book-ply", "4")
				# Scan.set("book-margin", "4")
				# Scan.set("ponder", "false")
				# Scan.set("threads", "1")
				# Scan.set("tt-size", "24")
				# Scan.set("bb-size", "6")
				# Scan.update()
				# Scan.init()
				
				# print(EGDB.probe(currentNode.state.board))
			# lg.logger_mcts.info('PLAYER TURN...%d', currentNode.state.playerTurn)
		
			maxQU = -99999

			if currentNode == self.root:
				epsilon = config.EPSILON
				nu = np.random.dirichlet([config.ALPHA] * len(currentNode.edges))
			else:
				epsilon = 0
				nu = [0] * len(currentNode.edges)

			Nb = 0
			for action, move, edge in currentNode.edges:
				Nb = Nb + edge.stats['N']

			for idx, (action, move, edge) in enumerate(currentNode.edges):
				U = self.cpuct * \
					((1-epsilon) * edge.stats['P'] + epsilon * nu[idx] )  * \
					np.sqrt(Nb) / (1 + edge.stats['N'])
					
				Q = edge.stats['Q']

				# lg.logger_mcts.info('action: %d (%d)... N = %d, P = %f, nu = %f, adjP = %f, W = %f, Q = %f, U = %f, Q+U = %f'
				# 	, action, action % 7, edge.stats['N'], np.round(edge.stats['P'],6), np.round(nu[idx],6), ((1-epsilon) * edge.stats['P'] + epsilon * nu[idx] )
				# 	, np.round(edge.stats['W'],6), np.round(Q,6), np.round(U,6), np.round(Q+U,6))

				if Q + U > maxQU:
					# print('2 ',end='')
					maxQU = Q + U
					simulationAction = action
					simulationEdge = edge

			# lg.logger_mcts.info('action with highest Q + U...%d', simulationAction)

			newState, value, done, _, _ = currentNode.state.takeAction(simulationAction,currentNode.count) #the value of the newState from the POV of the new playerTurn
			currentNode = simulationEdge.outNode
			breadcrumbs.append(simulationEdge)

			cc += 1
		# lg.logger_mcts.info('DONE...%d', done)

		return currentNode, value, done, breadcrumbs


	def backFill(self, leaf, value, breadcrumbs):
		
		# lg.logger_mcts.info('------DOING BACKFILL------')

		currentPlayer = leaf.state.playerTurn


		for edge in breadcrumbs:
			playerTurn = edge.playerTurn
			if playerTurn == currentPlayer:
				direction = 1
			else:
				direction = -1

			edge.stats['N'] = edge.stats['N'] + 1
			edge.stats['W'] = edge.stats['W'] + value * direction
			edge.stats['Q'] = edge.stats['W'] / edge.stats['N']

			# lg.logger_mcts.info('updating edge with value %f for player %d... N = %d, W = %f, Q = %f'
			# 	, value * direction
			# 	, playerTurn
			# 	, edge.stats['N']
			# 	, edge.stats['W']
			# 	, edge.stats['Q']
			# 	)

			# edge.outNode.state.render(lg.logger_mcts)
			
	def addNode(self, node):
		self.tree[node.id] = node

