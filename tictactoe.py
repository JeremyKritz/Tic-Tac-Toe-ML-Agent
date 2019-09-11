import numpy as np
import random
import gym
from keras.models import Sequential
from keras.layers import Dense, BatchNormalization
from keras.optimizers import SGD, RMSprop,Adam
from collections import deque
from keras.models import load_model


def drawBoard(board):

# This function prints out the board that it was passed.

	print('   |   |')
	print(' ' + board[0] + ' | ' + board[1] + ' | ' + board[2])
	print('   |   |')
	print('-----------')
	print('   |   |')
	print(' ' + board[3] + ' | ' + board[4] + ' | ' + board[5])
	print('   |   |')
	print('-----------')
	print('   |   |')
	print(' ' + board[6] + ' | ' + board[7] + ' | ' + board[8])
	print('   |   |')
	
def makeMove(board, letter, move):
	board[move] = letter

def isWinner(board, letter):

# Given a board and a player’s letter, this function returns True if that player has won.


	return ((board[0] == letter and board[1] == letter and board[2] == letter) or # across the top
	(board[3] == letter and board[4] == letter and board[5] == letter) or # across the middle
	(board[6] == letter and board[7] == letter and board[8] == letter) or # across the board
	
	(board[0] == letter and board[3] == letter and board[6] == letter) or # down the letterft side
	(board[1] == letter and board[4] == letter and board[7] == letter) or # down the middle	
	(board[2] == letter and board[5] == letter and board[8] == letter) or # down the right side
	
	(board[0] == letter and board[4] == letter and board[8] == letter) or # diagonal
	(board[2] == letter and board[4] == letter and board[6] == letter)) # diagonal

def isSpaceFree(board,space):
	if board[space] == '' or board[space] == ' ':
		return True
	else:
		return False
		
def createCopy(board):
	copy = []
	for i in board:
		copy.append(i)
	return copy
		
def chooseRandomMoveFromList(board, movesList):

	# Returns a valid move from the passed list on the passed board.

	# Returns None if there is no valid move.
	possibleMoves = []
	for i in movesList:
		if isSpaceFree(board, i):
			possibleMoves.append(i)
	if len(possibleMoves) != 0:
		return random.choice(possibleMoves)
	else:
		return None

def hardcodedAiMove(board,aiL):
	playerL = ''
	if aiL == 'X':
		playerL = 'O'
	else:
		playerL = 'X'
	# First, check if we can win in the next move
	for i in range(0, 9):
		copy = createCopy(board)

		if isSpaceFree(copy, i):
			makeMove(copy, aiL, i)

		if isWinner(copy, aiL):
			return i
	 # Check if the player could win on their next move, and block them.

	for i in range(0, 9):
		copy = createCopy(board)
		if isSpaceFree(copy, i):
			makeMove(copy, playerL, i)
		if isWinner(copy, playerL):
			return i			
	 # Try to take one of the corners, if they are free.
	move = chooseRandomMoveFromList(board, [0,1,2,3,4,5,6,7,8])
	return move
	
	
	
	#making the ai a bit worse intentionally to add variety
	'''
	move = chooseRandomMoveFromList(board, [0, 2, 6, 8])
	if move!= None:
		return move
	#take center
	if isSpaceFree(board,4):
		return 4
	return chooseRandomMoveFromList(board, [1, 3, 5, 7])
	'''
		
def createEmptyBoard():
	board = [' ',' ',' ',' ',' ',' ',' ',' ',' ']
	return board

def lettersToNumerical(board):
	numBoard = []
	for i in board:
		if i == ' ':
			numBoard.append(0)
		elif i == 'O':
			numBoard.append(-.5)
		else:
			numBoard.append(.5)
	return numBoard
		
def createTrainingData():
	print('hi')
	decay = .5
	xWins = 0
	draws = 0
	oWins = 0
	states = []
	player = ''
	labels = []
	numGames = 10000
	allMoves = [0,1,2,3,4,5,6,7,8]
	for i in range(numGames):
		print('Game ',i)
		gameStates = []
		gameLabels = []
		board = createEmptyBoard()
		gameOver = False
		xTurn = True
		gameResult = 0
		moves = 0
		while gameOver!= True:				
			#print('Moves',moves)		
			if xTurn:
				player = 'X'
			else:
				player = 'O'
			#print(' ',player, ' move')
			move = hardcodedAiMove(board,player)
			#print('move',move)
			#move = chooseRandomMoveFromList(board,allMoves)
			if move == None:
				#print('Game was a draw')
				draws+=1
				gameOver = True
				gameResult = .5
			else:
				makeMove(board,player,move)
				#print(gameStates)
				if player == 'X':
					#print('x moved')
					copy = []
					for i in board:
						copy.append(i)
					gameStates.append(copy)
					#print(gameStates)
					gameLabels.append(0) #a placeholder
				#drawBoard(board)      #uncomment to see the game
				if isWinner(board,player):
					#print(player, ' won')
					if player == 'X':
						xWins+=1
						gameResult = .95
					else:
						oWins+=1
						gameResult = -.95
					gameOver = True
			xTurn = not xTurn
			moves+=1
			#print('xT',xTurn)
		
		
		#At this point game is over, need to add to states list
		#a win is 1 a draw is 0 a loss is -1
		#if gameResult == -1:
			#print('loss')
		#print(gameStates)
		xTurns = len(gameStates)-1
		
		for x in range(0,len(gameStates)-1):
			#print(gameResult , decay , (x+1))
			gameLabels[xTurns-x] = gameResult * (decay ** (x))
			
		gameLabels[xTurns] = gameResult
		#print(gameLabels)
		for x in range(len(gameStates)):
			states.append(gameStates[x])
			labels.append(gameLabels[x])
	print('X,O,draw',xWins,oWins,draws)
	#for x in range(len(states)):
		#print(states[x],labels[x])
	return states, labels
						
def kerasStuff(states,labels):
	x_train = []
	for i in states:
		numB = lettersToNumerical(i)
		x_train.append(numB)
	x_train = np.asarray(x_train)
	x_train = x_train.astype('float32')
	y_train = np.asarray(labels)
	y_train = y_train.astype('float32')
	'''
	model = Sequential()
	model.add(Dense(9,activation = 'relu'))
	model.add(Dense(9,activation = 'relu'))
	model.add(Dense(5,activation = 'relu'))
	model.add(Dense(5,activation = 'relu'))
	model.add(BatchNormalization())
	model.add(Dense(3,activation = 'relu'))
	model.add(Dense(3,activation = 'relu'))	
	model.add(Dense(1,activation = 'linear'))
	
	#sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
	#RMSprop
	model.compile(loss='mean_squared_error', optimizer='Adam')
	'''
	model = load_model('tictactoeFinal3.h5')
	model.fit(x_train, y_train, epochs=20, batch_size=16)
	model.save('tictactoeFinal4.h5')
	
def testAgent():
	#model = load_model('tictactoe2.h5')
	model = load_model('tictactoeTrain2.h5')
	#winningBoard = ['X','O',' ', 'X',' ',' ','X',' ','O']
	#wB = [np.asarray(lettersToNumerical(winningBoard)).astype('float32')]
	#wB = np.transpose(wB)
	#print(wB)
	#npwB = []
	#for x in range(len(wB)):
		#npwB.append(wB[x])
		#print(wB,npwB)
	#print(model.predict(wB))
	sP = np.array([[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]])
	print('sp',sP)
	startPredict= model.predict(sP)
	winPredict = model.predict(np.array([[0.4, -0.4, 0.0, 0.4, 0.0, 0.0, 0.4, 0.0, -0.4]]))
	losePredict = model.predict(np.array([[-0.4, 0.4, 0.0, -0.4, 0.0, 0.0, -0.4, 0.4, 0.4]]))
	print(winPredict)
	print(losePredict)
	print(startPredict)	
	
def playAgent():
	model = load_model('tictactoeFinal3.h5')
	numGames = 200
	xWins = 0
	draws = 0
	oWins = 0


	allMoves = [0,1,2,3,4,5,6,7,8]
	for i in range(numGames):
		#print('Game ',i)

		board = createEmptyBoard()
		gameOver = False
		xTurn = True
		gameResult = 0
		moves = 0
		while gameOver!= True:
			#drawBoard(board)	
			#print('Moves',moves)		
			if xTurn:
				player = 'X'
			else:
				player = 'O'
			
			if player == 'X':
				#drawBoard(board)
				bestValue = -99999
				bestMove = None
				
				for i in range(0, 9):
					if isSpaceFree(board,i):
						predictFormat = np.array([[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]])
						copy = createCopy(board)
						makeMove(copy,'X',i)
						numCopy = np.array(lettersToNumerical(copy))
						for j in range(0,9):
							predictFormat[0][j] = numCopy[j]
						#print(predictFormat[0][1])
						#numCopy = [numCopy]
						#print(predictFormat)
						moveValue = model.predict(predictFormat)
						#print('Move ',i, 'value: ', moveValue)
						if moveValue > bestValue:
							bestValue = moveValue
							bestMove = i
				move = bestMove
			else:			
				#O will just be the AI
				move = hardcodedAiMove(board,player)
			#print('move',move)
			#move = chooseRandomMoveFromList(board,allMoves)
			if move == None:
				#print('Game was a draw')
				draws+=1
				gameOver = True
			else:
				makeMove(board,player,move)
				if isWinner(board,player):
					#print(player, ' won')
					if player == 'X':
						xWins+=1
					else:
						oWins+=1
					gameOver = True
			xTurn = not xTurn
			moves+=1
	print('X:',xWins,'O:',oWins,'Draws:',draws)
	
def randomPlayer():

	numGames = 200
	xWins = 0
	draws = 0
	oWins = 0


	allMoves = [0,1,2,3,4,5,6,7,8]
	for i in range(numGames):
		#print('Game ',i)

		board = createEmptyBoard()
		gameOver = False
		xTurn = True
		gameResult = 0
		moves = 0
		while gameOver!= True:
				
			#print('Moves',moves)		
			if xTurn:
				player = 'X'
			else:
				player = 'O'
			
			if player == 'X':
				move = chooseRandomMoveFromList(board,allMoves)
			else:			
				#O will just be the AI
				move = hardcodedAiMove(board,player)
			#print('move',move)
			#move = chooseRandomMoveFromList(board,allMoves)
			if move == None:
				#print('Game was a draw')
				draws+=1
				gameOver = True
			else:
				makeMove(board,player,move)
				if isWinner(board,player):
					#print(player, ' won')
					if player == 'X':
						xWins+=1
					else:
						oWins+=1
					gameOver = True
			xTurn = not xTurn
			moves+=1
	print('X:',xWins,'O:',oWins,'Draws:',draws)

def hardcodedAiPlayer():
	numGames = 200
	xWins = 0
	draws = 0
	oWins = 0


	allMoves = [0,1,2,3,4,5,6,7,8]
	for i in range(numGames):
		#print('Game ',i)

		board = createEmptyBoard()
		gameOver = False
		xTurn = True
		gameResult = 0
		moves = 0
		while gameOver!= True:
				
			#print('Moves',moves)		
			if xTurn:
				player = 'X'
			else:
				player = 'O'
			move = hardcodedAiMove(board,player)
			
			if move == None:
				#print('Game was a draw')
				draws+=1
				gameOver = True
			else:
				makeMove(board,player,move)
				if isWinner(board,player):
					#print(player, ' won')
					if player == 'X':
						xWins+=1
					else:
						oWins+=1
					gameOver = True
			xTurn = not xTurn
			moves+=1
	print('X:',xWins,'O:',oWins,'Draws:',draws)
	
	
board = createEmptyBoard()
drawBoard(board)
states = []
labels = []


#states, labels = createTrainingData()
#kerasStuff(states,labels)


#testAgent()
print('Agent:')
playAgent()
print('Random player')
randomPlayer()
