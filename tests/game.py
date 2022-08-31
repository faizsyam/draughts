import numpy as np
import logging

# import unittest
from draughts1 import *
import pandas as pd
import random

normal_move_df = pd.read_csv('normal_moves.csv')
king_move_df = pd.read_csv('king_moves.csv')

class Game:

	def __init__(self):	
		self.gameState = GameState(start_position())
		self.actionSpace = np.zeros((2,50), dtype=np.int)
		self.grid_shape = (10,10)
		self.input_shape = (4,10,10)
		self.name = 'draughts'
		self.state_size = len(self.gameState.binary)
		# self.action_size = 81
		self.action_size = 570

	def reset(self):
		self.gameState = GameState(start_position())
		return self.gameState

	def step(self, action):
		next_state, value, done = self.gameState.takeAction(action)
		self.gameState = next_state
		# self.currentPlayer = -self.currentPlayer
		info = None
		return ((next_state, value, done, info))

	def identities(self, state, actionValues):
		# FIX IDENTITIES
		identities = [(state.binary,state.playerTurn,actionValues)]
		
		# bb = state.board
		# bb.flip()
		# flipped_state = GameState(bb)

		# flipped_av = []
		# for move_id in range(570):

		# 	mv = king_move_df[king_move_df['move_id']==move_id]
		# 	from_ = mv['from'].iloc[0]
		# 	to_ = mv['to'].iloc[0]
			
		# 	from_ = 51 - from_
		# 	to_ = 51 - to_

		# 	new_id = king_move_df[(king_move_df['from']==from_)&(king_move_df['to']==to_)]['move_id'].iloc[0]
			
		# 	flipped_av.append(actionValues[new_id])

		# identities.append((flipped_state,actionValues))
		# identities.append((flipped_state,flipped_av))
		# text = print_position(state.board, False, True)
		# currentBoard = list(text[:-1])
		# currentBoard = np.reshape(currentBoard,(10,5))

		# flipped_board = state.board
		# flipped_board.flip()
		# identities.append((GameState(flipped_board), ))

		# currentAV1 = actionValues[:50]
		# currentAV2 = actionValues[50:]
		# currentAV1 = np.reshape(currentAV1,(10,5))
		# currentAV2 = np.reshape(currentAV2,(10,5))

		# currentBoard = np.flip(currentBoard,0)
		# currentAV1 = np.flip(currentAV1,0)
		# currentAV2 = np.flip(currentAV2,0)

		# board1_text = board_to_text(state.board,currentBoard)

		# if board1_text != text:
		# 	currentAV = list(np.reshape(currentAV1,(50))) + list(np.reshape(currentAV2,(50)))
		# 	identities.append((GameState(parse_position(board1_text)), currentAV))
		

		# currentBoard = np.flip(currentBoard,1)
		# currentAV1 = np.flip(currentAV1,0)
		# currentAV2 = np.flip(currentAV2,0)

		# board2_text = board_to_text(state.board,currentBoard)

		# if board2_text != board1_text:
		# 	currentAV = list(np.reshape(currentAV1,(50))) + list(np.reshape(currentAV2,(50)))
		# 	identities.append((GameState(parse_position(board2_text)), currentAV))

		# currentBoard = np.flip(currentBoard,0)
		# currentAV1 = np.flip(currentAV1,0)
		# currentAV2 = np.flip(currentAV2,0)

		# board3_text = board_to_text(state.board,currentBoard)

		# if (board3_text != board1_text)&(board3_text != board2_text):
		# 	currentAV = list(np.reshape(currentAV1,(50))) + list(np.reshape(currentAV2,(50)))
		# 	identities.append((GameState(parse_position(board3_text)), currentAV))

		return identities


class GameState():
	def __init__(self, board):
		self.board = board
		self.playerTurn = 1 if board.is_white_to_move() else -1
		self.binary = self._binary()
		self.id = self._convertStateToId()
		self.allowedActions = self._allowedActions()
		self.isEndGame = self._checkForEndGame()
		self.value = self._getValue()
		self.score = self._getScore()

	def _allowedActions(self):
		board = self.board
		# if not board.is_white_to_move():
			# board.flip()
		moves = generate_moves(board)
		return moves

	def _binary(self):

		return pos_to_numpy1(self.board)
		# text = print_position(self.board, False, True)

		# chr_player = 'o' if self.board.is_white_to_move() else 'x'
		# chr_other = 'x' if self.board.is_white_to_move() else 'o'

		# currentplayer_man = []
		# other_man = []
		# currentplayer_king = []
		# other_king = []

		# for i in text[:-1]:
		# 	currentplayer_man.append(1 if i==chr_player else 0)
		# 	other_man.append(1 if i==chr_other else 0)
		# 	currentplayer_king.append(1 if i==chr_player.upper() else 0)
		# 	other_king.append(1 if i==chr_other.upper() else 0)

		# position = np.concatenate((currentplayer_man, other_man, currentplayer_king, other_king), axis=0)
		# position = position.flatten()

		# return (position)

	def _convertStateToId(self):
		
		position = pos_to_numpy1(self.board)
		return ''.join(map(str,position))

		text = print_position(self.board, False, True)

		white_man = []
		black_man = []
		white_king = []
		black_king = []

		for i in text[:-1]:
			white_man.append(1 if i=='o' else 0)
			black_man.append(1 if i=='x' else 0)
			white_king.append(1 if i=='O' else 0)
			black_king.append(1 if i=='X' else 0)

		position = np.concatenate((white_man, black_man, white_king, black_king), axis=0)
		position = position.flatten()

		id = ''.join(map(str,position))

		return id

	def _checkForEndGame(self):
		return self.board.is_end()


	def _getValue(self):
		# This is the value of the state for the current player
		# i.e. if the previous player played a winning move, you lose
		# for x,y,z,a in self.winners:
		# 	if (self.board[x] + self.board[y] + self.board[z] + self.board[a] == 4 * -self.playerTurn):
		if self.board.is_end():
			# print(self.board.turn())
			# display_position(self.board)
			# print('turn: ',self.board.turn())
			# print('res: ',self.board.result(self.board.turn()))
			turn = 1 if self.board.turn()==Side.White else -1
			return (turn, turn, 1)
			# return (self.playerTurn, self.playerTurn, 1)
		return (0, 0, 0)


	def _getScore(self):
		tmp = self.value
		return (tmp[1], tmp[2])


	def takeAction(self, action):
		# print('in')
		board = self.board
		# if not self.board.is_white_to_move():
			# board.flip()
		newpos = play_forced_moves(board.succ(action))
		count = 0
		# print('in2')
		while newpos.is_capture():
			count += 1
			moves = generate_moves(newpos)
			try:
				newpos = play_forced_moves(newpos.succ(moves[random.randint(0,len(moves)-1)]))
			except:
				display_position(newpos)
				newpos = play_forced_moves(newpos.succ(moves[random.randint(0,len(moves)-1)]))

			
		# if not self.board.is_white_to_move():
			# newpos.flip()
		
		newState = GameState(newpos)
		# print('outs: ', count)

		value = 0
		done = 0

		# if ((newpos.white_king_count() > 0) | (newpos.black_king_count() > 0)) & (self.board.white_king_count() == 0) & (self.board.black_king_count() == 0):

			# print('XXXXX ',print_move(action, self.board), count)
			# display_position(self.board)
			# display_position(newpos)

		if newpos.is_end():
			value = newState.value[0]
			# newState.playerTurn = -newState.playerTurn
			done = 1

		return (newState, value, done) 


	def render(self, logger):		
		# text = print_position(self.board, False, True)
		# for r in range(10):
		# 	logger.info("".join([text[5*r : (5*r + 5)]]))
		# logger.info('--------------')
		a=1

def board_to_text(pos, currentBoard):
	text = ""
	for x in currentBoard:
		for y in x:
			text += y
	text += 'W' if pos.is_white_to_move() else'B'
	return text


