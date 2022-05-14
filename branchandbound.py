
import numpy as np
import sys
import math

oristdout = sys.stdout
sys.stdout = open('./output.txt', 'w+')


# Taking input from the input file
file = open('./input.txt', 'r')

AA = np.zeros((0,0))
matrixb = []
matrixc = []
flagA = -1
currentMat = 'A'
ansflag = -1
precision = 0.000000001

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
				AA = np.zeros((1, len(words)), np.float)
			else:
				AA = np.append(AA, np.zeros((1, len(words)), np.float), axis = 0)
			col = 0
			row = AA.shape[0] - 1
			for word in words:
				AA[row][col] = float(word)
				col += 1

mapperToEdges = np.zeros(AA.shape, np.int)
mapval = 0

if AA.shape[0]==2:
	print(f"{AA[1][0]*2}")
	print(f"0 1 1 0")
	print(1)
	exit(0)

for i in range(AA.shape[0]):
	for j in range(AA.shape[1]):
		if i==j:
			mapperToEdges[i][j] = -1
		if i>=j:
			mapperToEdges[i][j] = mapperToEdges[j][i]
			continue
		mapperToEdges[i][j] = mapval
		matrixc.append(AA[i][j])
		mapval += 1

matrixA = np.zeros((0,mapval), np.float)

matrixc = np.array(matrixc, np.float)
matrixc = np.reshape(matrixc, (len(matrixc),1))

for i in range(AA.shape[0]):
	temaa = np.zeros((1, mapval), np.float)
	for j in range(AA.shape[1]):	
		if mapperToEdges[i][j]!=-1:
			temaa[0][mapperToEdges[i][j]] = 1
	matrixA = np.append(matrixA, temaa, axis = 0)
	matrixA = np.append(matrixA, -temaa, axis = 0)
	matrixb.append(2)
	matrixb.append(-2)

for i in range(matrixc.shape[0]):
	temaa = np.zeros((1, mapval), np.float)
	temaa[0][i] = 1
	matrixA = np.append(matrixA, temaa, axis = 0)
	matrixb.append(1)

for i in range(int(2**AA.shape[0])-1):
	if i == 0:
		continue
	listA = []
	listB = []
	temi = i
	for jj in range(AA.shape[0]):
		if temi%2==1:
			listA.append(jj)
		else:
			listB.append(jj)
		temi//=2
	
	temaa = np.zeros((1, mapval), np.float)

	for jj in listA:
		for kk in listB:
			temaa[0][mapperToEdges[jj][kk]] = 1
	# print(i)
	# print(-temaa)
	# print(listA)
	matrixA = np.append(matrixA, -temaa, axis = 0)
	matrixb.append(-2)

matrixb = np.array(matrixb, np.float)
matrixb = np.reshape(matrixb, (len(matrixb),1))



# Verification of the input matrices
exns = matrixA.shape[1]
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

	print(f"Minimum Value: {updatedA[-1][-1]}")
	ans = np.zeros(exns)
	for i in range(updatedA.shape[0]-1):
		if basicvector[i] < exns:
			ans[basicvector[i]] = updatedA[i][updatedA.shape[1]-1]
	print("Optimal basic vector: ")
	for i in range(ans.shape[0]):
		print(ans[i], end=" ")
	print("\n")
	print(f"Nodes Explored: {totalIterations}")


def integrityConstraints(A, basicvector):
	global exns
	for i in range(A.shape[0]-1):
		if checkInteger(A[i][-1]) == 0 and basicvector[i] < exns:
			return 0
	return 1

def basisvectorInA(updatedA, basicvector):
	global exns
	ans = np.zeros((updatedA.shape[1]-1), np.float)

	for i in range(updatedA.shape[0]-1):
		ans[basicvector[i]] = updatedA[i][updatedA.shape[1]-1]
	return ans

totalIterations = 0

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


def branchBound(matrixA, matrixb):
	global ansflag
	global exns
	S = []
	S.append([matrixA, matrixb])
	optimalcost = "inf"
	optimalvector = []
	Failuremsg = "Infeasible"
	global totalIterations
	while len(S) > 0:
		totalIterations += 1
		[curMatA , curMatb] = S.pop(0)
		# S.pop()
		updatedA, basicvector = constructMatrices(np.copy(curMatA), np.copy(curMatb))
		convertFunctionalIntoCorrectForm(updatedA, np.copy(basicvector))
		temperoryA = phase1(updatedA, basicvector)
		if ansflag == 1:
			if integrityConstraints(temperoryA, basicvector) == 1:
				if optimalcost == "inf" or optimalcost > temperoryA[-1][-1]:
					optimalcost = temperoryA[-1][-1]
					optimalvector = basisvectorInA(temperoryA, basicvector)
			else:
				if optimalcost != "inf" and temperoryA[-1][-1] >= optimalcost:
					continue
				temvector = basisvectorInA(temperoryA, basicvector)
				indrow = -1
				for i in range(exns):
					if checkInteger(temvector[i]) == 0:
						indrow = i
						break

				if indrow == -1:
					continue
				temind = -1
				for i in range(len(basicvector)):
					if basicvector[i] == indrow:
						temind = i
						break
				
				actualvar = indrow
				indrow = temind

				rowvalue = temperoryA[indrow][-1]
				rowTomodify = np.copy(temperoryA[indrow])
				rowTomodify = np.zeros(rowTomodify.shape)
				rowTomodify[actualvar] = 1
				# print(rowTomodify)
				rowTomodify[-1] = math.floor(rowvalue)
				newMatA, newMatB = newRowInA(np.copy(curMatA), np.copy(curMatb), rowTomodify)
				S.append([newMatA, newMatB])
				# print("HIHIH")
				# print(newMatA)
				# print(newMatB)
				# print(rowTomodify)
				rowTomodify = np.copy(temperoryA[indrow])
				rowTomodify = np.zeros(rowTomodify.shape)
				rowTomodify[actualvar] = -1
				rowTomodify[-1] = -math.ceil(rowvalue)
				newMatA, newMatB = newRowInA(np.copy(curMatA), np.copy(curMatb), rowTomodify)
				S.append([newMatA, newMatB])
				# print("HIHIH")
				# print(newMatA)
				# print(newMatB)
				# print(rowTomodify)
	

		else:
			if temperoryA == "Unbounded":
				Failuremsg = "Unbounded"


	if optimalcost != "inf":
		# print(f"Min Cost: {optimalcost}")
		# print(f"Best Vector: {optimalvector[0:exns]}")
		# print(f"Total Iterations: {totalIterations}")
		print(f"{optimalcost}")
		vvvv = 0
		ansmat = np.zeros(AA.shape, np.int)
		global mapperToEdges
		adjmat = np.zeros(AA.shape, np.int)
		
		for i in range(AA.shape[0]):
			for j in range(AA.shape[1]):
				if int(round(optimalvector[mapperToEdges[i][j]])) == 1:
					adjmat[i][j] = 1

		curnode = 0
		visited = np.zeros((AA.shape[0]))
		visited[0] = 1

		while(1):
			prenode = curnode
			for i in range(AA.shape[0]):
				if adjmat[curnode][i] == 1 and visited[i]==0:
					ansmat[curnode][i] = 1
					visited[i] = 1
					curnode = i
					break
			if prenode == curnode:
				break
		ansmat[curnode][0] = 1

		for i in range(AA.shape[0]):
			for j in range(AA.shape[1]):
				print(f"{ansmat[i][j]}", end=" ")
				# if i==j:
				# 	print(f"{i}, {j}: 0")
				# elif i<j:
				# 	print(f"{i}, {j}: {optimalvector[vvvv]}")
				# 	vvvv+=1
				# else:
				# 	print(f"{i}, {j}: 0")

		print()
		print(f"{totalIterations}")

	else:
		print(Failuremsg)
		print(totalIterations)

branchBound(matrixA, matrixb)


# print(matrixA)
# print(matrixA.shape)


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
