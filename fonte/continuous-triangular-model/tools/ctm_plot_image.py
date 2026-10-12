
import csv	
import sys

from PIL import Image

#class Coord:
#    def __init__(self, x, y):
#        self.position = Coord(x,y) #Coord class created elsewhere
#
#class MyObject:
#    def __init__(self, x, y):
#        self.position = Coord(x,y) #Coord class created elsewhere

span = int(sys.argv[2])

class MyCoordList:
	def __init__(self):
		self.xList = []
		self.yList = []

	def sum(self, inputVar):
		eps = 1e-9
		inputXMin = self.xList[0]
		inputXMax = self.xList[len(self.xList) - 1]
#		print('--> ' + str(inputXMin))
#		print('--> ' + str(inputXMax))
		steps = span
		inputXStep = (inputXMax - inputXMin) / steps
		cnt = 0
		for i in range(0, steps):
			for j in range(0, i):
				for k in range(j, j + steps - i):
					l = 0
					acc = 0
					while self.xList[l] < (j - 1) * inputXStep + inputXMin + eps:
						l = l + 1
					while l < len(self.xList) and self.xList[l] <= (j + steps - i) * inputXStep + inputXMin + eps:
						acc = acc + self.yList[l]
						l = l + 1
					#print(cnt)
					cnt += 1
				inputVar[i][j] = acc
			print('i: = ' + str(i))
		return


	def normalize(self, inputVar):
		inputYMin = float("inf")
		inputYMax = -float("inf")
		for i in range(0, span):
			for j in range(0, span):
				inputYMin = min(inputYMin, inputVar[i][j])
				inputYMax = max(inputYMax, inputVar[i][j])
#		print('**** -> ' + str(inputYMin))
#		print('**** -> ' + str(inputYMax))
		for i in range(0, span):
			for j in range(0, span):
				inputVar[i][j] = inputVar[i][j] / (inputYMax - inputYMin) * 256
		return

# PIL accesses images in Cartesian co-ordinates, so it is Image[columns, rows]
img = Image.new( 'RGB', (span, span), "black") # create a new black image
pixels = img.load() # create the pixel map

a = MyCoordList()

with open(sys.argv[1]) as csvfile:
	myReader = csv.reader(csvfile)

#	for row in myReader:
#		print(row)

	for row in myReader:
#		print('-> ' + row[0])
#		print('-> ' + row[1])
		a.xList.append(float(row[0]))
		a.yList.append(float(row[1]))

#	print(a.xList)
#	print(a.yList)


#	for i in a.xList:
#		print(i)

#	print('end')

#	for i in a.yList:
#		print(i)

#	print('end')

#		print(', '.join(row))


w, h = span, span;
matrix = [[0 for x in range(w)] for y in range(h)] 

a.sum(matrix)

#for i in range(len(matrix)):    # for every col:
#	print(matrix[i])

a.normalize(matrix)

#for i in range(img.size[0]):    # for every col:
#    for j in range(img.size[1]):    # For every row
#        pixels[i,j] = (i, i, i) # set the colour accordingly


for i in range(len(matrix)):    # for every col:
	print(matrix[i])

offset = span / 2

for i in range(len(matrix)):    # for every col:
	for j in range(int(offset - i / 2), int(offset + i / 2)):    # For every row
		pixels[j, i] = (30 + int(matrix[i][j - int(offset - i / 2)]), 30 + int(matrix[i][j - int(offset - i / 2)]), 30 + int(matrix[i][j - int(offset - i / 2)])) # set the colour accordingly
#		pixels[j, i] = (255, 255, 255) # set the colour accordingly
#		pixels[i, j] = (255, 255, 255) # set the colour accordingly

img.show()
