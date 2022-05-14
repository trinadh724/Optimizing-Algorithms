import numpy as np
import sys
import math

oristdout = sys.stdout
sys.stdout = open('./output.txt', 'w+')


# Taking input from the input file
file = open('./input.txt', 'r')

# Taking input from the input file
# file = open(sys.stdin, 'r')

matrixA = np.zeros((0,0))
matrixb = np.zeros((0,0))
matrixc = np.zeros((0,0))
flagA = -1
currentMat = 'A'
ansflag = -1
precision = 0.000000001
totalIterations = 0
for line in file:
	words = line.split()
	if currentMat == 'A':
		if words[0] == "start":
			continue
		elif words[0] == "end":
			currentMat = 'b'
			continue
		else:
			if flagA == -1:
				flagA = 1
				# THe precision can be increased based on how much we want
				matrixA = np.zeros((1, len(words)), np.float)
			else:
				matrixA = np.append(matrixA, np.zeros((1, len(words)), np.float), axis = 0)
			col = 0
			row = matrixA.shape[0] - 1
			for word in words:
				matrixA[row][col] = float(word)
				col += 1

	elif currentMat == 'b':
		if words[0] == "start":
			continue
		elif words[0] == "end":
			currentMat = 'c'
			continue
		else:
			matrixb = np.zeros((len(words), 1), np.float)
			row = 0
			for word in words:
				matrixb[row][0] = float(word)
				row += 1
	else:
		if words[0] == "start":
			continue
		elif words[0] == "end":
			currentMat = 'd'
			continue
		else:
			matrixc = np.zeros((len(words), 1), np.float)
			row = 0
			for word in words:
				matrixc[row][0] = float(word)
				row += 1

# Verification of the input matrices
exns = matrixA.shape[1]
matrixc = -matrixc
# print("Matrix A: ")
# print(matrixA)

# print("Matrix b: ")
# print(matrixb)

# print("Matrix c: ")
# print(matrixc)


# Modifying matrix A to add Slack variables

artificialVariablesStarts = 0

def constructMatrices(matrixA, matrixb, basicvector=[], indexvector=[]):
	matrixA = np.copy(matrixA)
	matrixb = np.copy(matrixb)
	basicvector = []
	indexvector = []
	global artificialVariablesStarts
	for i in range(matrixA.shape[0]):
		temcol = np.zeros((matrixA.shape[0], 1), np.float)
		if matrixb[i][0] >= 0:
			basicvector.append(matrixA.shape[1])
			indexvector.append(i)

		temcol[i][0] = 1 
		matrixA = np.append(matrixA, temcol, axis = 1)

	phase1c = np.zeros((matrixA.shape[1], 1), np.float)
	artificialVariablesStarts = matrixA.shape[1]-1

	for i in range(matrixA.shape[0]):
		if matrixb[i][0] >= 0:
			continue

		basicvector.append(matrixA.shape[1])
		indexvector.append(i)
		temcol = np.zeros((matrixA.shape[0], 1), np.float)
		phase1c = np.append(phase1c, np.array([[-1]], np.float), axis = 0)
	
		temcol[i][0] = -1 
		matrixA = np.append(matrixA, temcol, axis = 1)
		matrixA[i, :] = -matrixA[i, :]
		matrixb[i][0] = -matrixb[i][0]

# print("Basic vectors")
# print(basicvector)
	bbasic = []
	for i in range(len(basicvector)):
		bbasic.append(0)
	for i in range(len(basicvector)):
		bbasic[indexvector[i]] = basicvector[i]

	basicvector = bbasic.copy()


	updatedA = np.append(matrixA, phase1c.T, axis = 0)
	updatedb = np.append(matrixb, np.array([[0]]), axis = 0)
	updatedA = np.append(updatedA, updatedb, axis = 1)

	# print(basicvector)
	return updatedA, basicvector


# Now the code for Actual Simplex algorithm starts

# This function makes sure that ith column in rowA is 0 using linear combintion of rowA with rowB
def linearCombination(rowA, rowB, i):
	rowA = np.copy(rowA)
	rowB = np.copy(rowB)
	# print(rowA.shape)
	rowA = np.reshape(rowA, (rowA.shape[0]))
	rowB = np.reshape(rowB, (rowB.shape[0]))

	if abs(rowB[i]) > precision:
		rowA = rowA - (rowA[i]/rowB[i]) * rowB
	return rowA


def convertFunctionalIntoCorrectForm(A, vectorbasic):
	row = 0
	# print(vectorbasic)
	for i in vectorbasic:
		A[A.shape[0]-1] = linearCombination(A[A.shape[0]-1], A[row], i)
		# print(A)
		row += 1
	return A

def basisUpdation(A, entering, rowleaving, vectorbasic):
	A[rowleaving] = A[rowleaving]/A[rowleaving][entering]
	for i in range(A.shape[0]):
		if i == rowleaving:
			continue
		else:
			A[i] = linearCombination(A[i], A[rowleaving], entering)
	vectorbasic[rowleaving] = entering


def enteringVariable(A):
	for i in range(A.shape[1]-1):
		if A[A.shape[0]-1][i] > 0 and abs(A[A.shape[0]-1][i]) > precision:
			return i
	return -1

def leavingvariable(A, entering):
	ratio = -1
	for i in range(A.shape[0]-1):
		if A[i][entering] > 0 and abs(A[i][entering]) > precision:
			if(ratio == -1):
				ratio = A[i][A.shape[1]-1]/A[i][entering]
			ratio = min(ratio , A[i][A.shape[1]-1]/A[i][entering])

	for i in range(A.shape[0]-1):
		if A[i][entering] > 0 and abs(A[i][entering]) > precision:
			if ratio == A[i][A.shape[1]-1]/A[i][entering]:
				return i

	return -1



def phase2Aupdation(A, vectorbasic):
	phase2c = np.zeros((A.shape[1], 1), np.float)
	phase2c[0 : matrixc.shape[0] , : ] = -matrixc
	A[A.shape[0]-1, :] = phase2c.T
	convertFunctionalIntoCorrectForm(A, vectorbasic)

# print(artificialVariablesStarts)
def phase2EnteringVariable(A):
	for i in range(artificialVariablesStarts+1):
		if A[A.shape[0]-1][i] > 0 and abs(A[A.shape[0]-1][i]) > precision:
			return i
	return -1

def phase2Leavingvariable(A, entering, basicvector):
	ratio = -1
	
	for i in range(A.shape[0]-1):
		if basicvector[i] > artificialVariablesStarts:
			if A[i][entering] < 0 and abs(A[i][entering]) > precision:
				return i


	for i in range(A.shape[0]-1):
		if A[i][entering] > 0 and abs(A[i][entering]) > precision:
			if(ratio == -1):
				ratio = A[i][A.shape[1]-1]/A[i][entering]
			ratio = min(ratio , A[i][A.shape[1]-1]/A[i][entering])

	for i in range(A.shape[0]-1):
		if A[i][entering] > 0 and abs(A[i][entering]) > precision:
			if ratio == A[i][A.shape[1]-1]/A[i][entering]:
				return i

	return -1

def phase2(A, vectorbasic):
	# print("Phase2")
	# print(A)
	global ansflag
	# print(vectorbasic)
	while phase2EnteringVariable(A) != -1:
		entering = phase2EnteringVariable(A)
		leaving = phase2Leavingvariable(A, entering, vectorbasic)
		if leaving == -1:
			ansflag = -1
			return "Unbounded"
		basisUpdation(A, entering, leaving, vectorbasic)
		# print("\n\n")
		# print(A)
		# print(vectorbasic)
	# print(ansflag)

	# print(ansflag)
	ansflag = 1
	return A
	

def phase1(A, vectorbasic):
	# print("Phase1")
	# print(A)
	# print(vectorbasic)
	global ansflag
	while enteringVariable(A) != -1:
		entering = enteringVariable(A)
		leaving = leavingvariable(A, entering)
		if leaving == -1:
			ansflag = -1
			return "Unbounded"
		basisUpdation(A, entering, leaving, vectorbasic)
		# print("\n\n")
		# print(A)
		# print(vectorbasic)

	if abs(A[-1][-1]) > precision:
		ansflag = -1
		return "Infeasible"
	phase2Aupdation(A, vectorbasic)
	return phase2(A, vectorbasic)

def checkInteger(num):
	num = abs(num)
	if num - math.floor(num) < precision:
		return 1
	elif math.ceil(num) - num < precision:
		return 1
	return 0

def printans(updatedA, basicvector):
	global exns
	global totalIterations
	print(f"{-updatedA[-1][-1]}")
	ans = np.zeros(exns)
	for i in range(updatedA.shape[0]-1):
		if basicvector[i] < exns:
			ans[basicvector[i]] = updatedA[i][updatedA.shape[1]-1]
	for i in range(ans.shape[0]):
		print(int(round(ans[i])), end=" ")
	print(f"\n{totalIterations-1}")

def newRowInA(matrixA, matrixb, rowTomodify):
	
	for i in range(matrixA.shape[0]):
		temrow = np.zeros(rowTomodify.shape, np.float)
		for j in range(matrixA.shape[1]):
			temrow[j] = matrixA[i][j]
		temrow[i + matrixA.shape[1]] = 1
		temrow[-1] = matrixb[i][0]
		rowTomodify = linearCombination(rowTomodify, temrow , i+matrixA.shape[1])


	addrow = np.zeros((matrixA.shape[1]), np.float)
	for i in range(addrow.shape[0]):
		addrow[i] = rowTomodify[i]
	addrow = np.reshape(addrow, (1, addrow.shape[0]))
	matrixA = np.append(matrixA, addrow, axis = 0)
	# print(matrixb)
	matrixb = np.append(matrixb, np.array([[rowTomodify[-1]]]), axis=0)
	return matrixA, matrixb	



def gomoraCut(matrixA, matrixb):

	updatedA, basicvector = constructMatrices(matrixA, matrixb)
	# print("gomoraCut")
	# print(updatedA)
	# print(basicvector)
	convertFunctionalIntoCorrectForm(updatedA, np.copy(basicvector))
	temperoryA = phase1(updatedA, basicvector)
	# print("Passed")
	global ansflag
	global exns
	global totalIterations
	if ansflag == -1:
		print(temperoryA)
		print(totalIterations-1)
		return -1, temperoryA, temperoryA

	cutrow = -1
	for i in range(temperoryA.shape[0]-1):
		curnum = temperoryA[i][-1]
		if checkInteger(curnum)==0 and basicvector[i] < exns:
			cutrow = i
			break
	if cutrow == -1:
		printans(temperoryA, basicvector)
		return 0, temperoryA, temperoryA

	rowTomodify = np.copy(temperoryA[cutrow])
	rowTomodify = np.floor(rowTomodify)
	# Experiment Trinadh Mon 11: 43AM
	# counter = 0
	# for i in range(rowTomodify.shape[0]-1):
	# 	if abs(rowTomodify[i]-temperoryA[cutrow][i]) > precision:
	# 		counter += 1
	# if counter == 0:

	matrixA, matrixb = newRowInA(matrixA, matrixb, rowTomodify)
	return 1, matrixA, matrixb

def cutAlgo(matrixA, matrixb):
	ff = 1
	global totalIterations 
	while(1):
		totalIterations += 1
		ff , matrixA, matrixb = gomoraCut(matrixA, matrixb)
		if ff != 1:
			break
	return 0

cutAlgo(matrixA, matrixb)

sys.stdout = oristdout


















# updatedA, basicvector = constructMatrices(matrixA, matrixb)
# convertFunctionalIntoCorrectForm(updatedA, np.copy(basicvector))
# vv = phase1(updatedA, basicvector)

# # print(ansflag)
# print("Final ANSWER")
# if ansflag == -1:
# 	print(vv)
# else:
# 	print(f"Minimum Value: {updatedA[-1][-1]}")
# 	ans = np.zeros(exns)
# 	for i in range(updatedA.shape[0]-1):
# 		if basicvector[i] < exns:
# 			ans[basicvector[i]] = updatedA[i][updatedA.shape[1]-1]
# 	print("Optimal basic vector: ")
# 	for i in range(ans.shape[0]):
# 		print(ans[i], end=" ")
# 	print("\n")
