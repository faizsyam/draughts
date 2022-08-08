import numpy as np
import logging

# import unittest
from draughts1 import *
import pandas as pd

class Game:

	def __init__(self):	
		self.gameState = GameState(start_position())
		self.actionSpace = np.zeros((2,50), dtype=np.int)
		self.grid_shape = (5,10)
		self.input_shape = (4,5,10)
		self.name = 'draughts'
		self.state_size = len(self.gameState.binary)
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
		identities = [(state,actionValues)]
		
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
		moves = generate_moves(self.board)
		return moves

	def _binary(self):

		text = print_position(self.board, False, True)

		chr_player = 'o' if self.board.is_white_to_move() else 'x'
		chr_other = 'x' if self.board.is_white_to_move() else 'o'

		currentplayer_man = []
		other_man = []
		currentplayer_king = []
		other_king = []

		for i in text[:-1]:
			currentplayer_man.append(1 if i==chr_player else 0)
			other_man.append(1 if i==chr_other else 0)
			currentplayer_king.append(1 if i==chr_player.upper() else 0)
			other_king.append(1 if i==chr_other.upper() else 0)

		position = np.concatenate((currentplayer_man, other_man, currentplayer_king, other_king), axis=0)
		position = position.flatten()

		return (position)

	def _convertStateToId(self):

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
			return (self.playerTurn, self.playerTurn, 1)
		return (0, 0, 0)


	def _getScore(self):
		tmp = self.value
		return (tmp[1], tmp[2])


	def takeAction(self, action):
		newpos = self.board.succ(action)
		newState = GameState(newpos)

		value = 0
		done = 0

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

def init_move_df():
	board = [[0,1,0,2,0,3,0,4,0,5]
        ,[6,0,7,0,8,0,9,0,10,0]
        ,[0,11,0,12,0,13,0,14,0,15]
        ,[16,0,17,0,18,0,19,0,20,0]
        ,[0,21,0,22,0,23,0,24,0,25]
        ,[26,0,27,0,28,0,29,0,30,0]
        ,[0,31,0,32,0,33,0,34,0,35]
        ,[36,0,37,0,38,0,39,0,40,0]
        ,[0,41,0,42,0,43,0,44,0,45]
        ,[46,0,47,0,48,0,49,0,50,0]]

	moves = []
	idx = 0

	for i in range(10):
		for j in range(10):
			if board[i][j]!=0:
				start = board[i][j]
				a = i + 1
				b = j + 1
				while (a<10)&(b<10):
					end = board[a][b]
					moves.append([idx,start,end])
					idx+=1
					a+=1
					b+=1
				a = i + 1
				b = j - 1
				while (a<10)&(b>=0):
					end = board[a][b]
					moves.append([idx,start,end])
					idx+=1
					a+=1
					b-=1
				a = i - 1
				b = j + 1
				while (a>=0)&(b<10):
					end = board[a][b]
					moves.append([idx,start,end])
					idx+=1
					a-=1
					b+=1
				a = i - 1
				b = j - 1
				while (a>=0)&(b>=0):
					end = board[a][b]
					moves.append([idx,start,end])
					idx+=1
					a-=1
					b-=1

	mdf = pd.DataFrame(moves,columns=['index','start','end'])

	return mdf

move_df = init_move_df()