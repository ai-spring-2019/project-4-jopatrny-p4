"""
Julia Opatrny
Project 4: Classification with Neural Networks
Usage: python3 project3.py DATASET.csv

This program creates a neural network, and does forward propagation and back propagation
on the layers and hidden layers to adjust weights and predict outcomes.

To run: create a NeuralNetwork object by passing a list containing
the number of nodes per layer.
For instance, to create a 3x6x3 Neural Network:
net = NeuralNetwork([3, 6, 3])

Run front propagation and back propagation on the inputs however many times you like like so:
for example in training:
	frontpropagation(net, example, [3, 6, 3])
	backpropagation(net, example, [3, 6, 3])
"""

import csv, sys, random, math, copy

def read_data(filename, delimiter=",", has_header=True):
    """Reads datafile using given delimiter. Returns a header and a list of
    the rows of data."""
    data = []
    header = []
    with open(filename) as f:
        reader = csv.reader(f, delimiter=delimiter)
        if has_header:
            header = next(reader, None)
        for line in reader:
            example = [float(x) for x in line]
            data.append(example)

        return header, data

def convert_data_to_pairs(data, header):
    """Turns a data list of lists into a list of (attribute, target) pairs."""
    pairs = []
    for example in data:
        x = []
        y = []
        for i, element in enumerate(example):
            if header[i].startswith("target"):
                y.append(element)
            else:
                x.append(element)
        pair = (x, y)
        pairs.append(pair)
    return pairs

def dot_product(v1, v2):
    """Computes the dot product of v1 and v2"""
    sum = 0
    for i in range(len(v1)):
        sum += v1[i] * v2[i]
    return sum

def logistic(x):
    """Logistic / sigmoid function"""
    try:
        denom = (1 + math.e ** -x)
    except OverflowError:
        return 0.0
    return 1.0 / denom

def accuracy(nn, pairs):
    """Computes the accuracy of a network on given pairs. Assumes nn has a
    predict_class method, which gives the predicted class for the last run
    forward_propagate. Also assumes that the y-values only have a single
    element, which is the predicted class.
    Optionally, you can implement the get_outputs method and uncomment the code
    below, which will let you see which outputs it is getting right/wrong.
    Note: this will not work for non-classification problems like the 3-bit
    incrementer."""

    true_positives = 0
    total = len(pairs)

    for (x, y) in pairs:
        forwardpropagate(nn, (x, y), nn.nodes)
       	print("x", y)
        class_prediction = nn.predict_class()
        if class_prediction != y[0]:
            true_positives += 1

        # outputs = nn.get_outputs()
        # print("y =", y, ",class_pred =", class_prediction, ", outputs =", outputs)

    return 1 - (true_positives / total)

################################################################################
### Neural Network code goes here
class NeuralNetwork:
	def __init__(self, nodes):
		self.layers = len(nodes)
		#print("NODES", nodes) 
		self.nodes = nodes
		self.wmatrix = []
		self.injmatrix = []
		self.ajmatrix = []
		self.wmatrix.append([])
		self.errorj = []
		self.errori = []
		for i in range(1, self.layers):
			layer = []
			for j in range(nodes[i - 1]):
				row = []
				for k in range(nodes[i]):
					row.append(random.random())
				layer.append(row)
			self.wmatrix.append(layer)
		self.layermatrix = []
		self.amatrix = []
		for l in range(self.layers):
			self.layermatrix.append([])
		for n in nodes:
			row = []
			for i in range(n):
				row.append([])
			self.amatrix.append(copy.deepcopy(row))
			self.injmatrix.append(copy.deepcopy(row))
			self.ajmatrix.append(copy.deepcopy(row))
			self.errorj.append(copy.deepcopy(row))
			self.errori.append(copy.deepcopy(row))
		#print("&&&&&&&&&&&&", self.wmatrix)
		#print("LENGTH", len(self.injmatrix))
		#print(self.layermatrix)
		#print(self.amatrix)
		#print(self.injmatrix)

	def show(self):
		for row in self.wmatrix:
			for col in row:
				print(col)

	def predict_class(self):
		ajsum = 0
		for item in self.ajmatrix[-1]:
			ajsum += item
		prediction = (ajsum/len(self.ajmatrix[-1]))
		print("PREDICTION", round(prediction), prediction)
		return round(prediction)


def forwardpropagate(network, example, layers):
	""" Propagates inputs forward to compute outputs"""
	nodes = network.nodes

	for i in range(len(example[0])):
		network.ajmatrix[0][i] = example[0][i]

	for l in range(1, len(layers)):
		for j in range(layers[l]):
			tempinj = 0
			for n in range(len(network.wmatrix[l])):
				tempinj += network.ajmatrix[l-1][n] * network.wmatrix[l][n][j]
			#print("REG:", tempinj, "LOG", logistic(tempinj))
			network.injmatrix[l][j] = tempinj
			network.ajmatrix[l][j] = logistic(tempinj)


	"""
	for l in range(1, len(layers)):
		#print("LENGTH", len(layers))
		#for each layer l in 2 to L
		#print("**l", l)
		for j in range(nodes[l]):
			#print("HI", nodes[l])
			#for each node j in layer l
			#self.injmatrix[l][j]
			tempinj = 0
			for r in range(len(network.wmatrix[l])):
				#print("LENGTH", len(network.wmatrix[l]))
				#print("layer", l, "node", j, r)
				#print(network.ajmatrix, "***", network.ajmatrix[l-1][r])
				#print(network.wmatrix[l][r][j])
				tempinj += network.wmatrix[l][r][j] * network.ajmatrix[l-1][r]
			print("REG:", tempinj, "LOG", logistic(tempinj))
			network.injmatrix[l][j] = tempinj
			network.ajmatrix[l][j] = logistic(tempinj)
	"""
				#tempinj += network.wmatrix[r][j] # * network.ajmatrix[l-1][j]
				#print(tempinj)
	"""
			tempinj = 0
			#print("j", j)
			#print("NET", network.wmatrix[l-1])
			#print("####", network.wmatrix)
			for col in network.wmatrix[l]:
				for num in col:
					#print("**********")
					#print(network.layermatrix[0])
					#print("---NUM", num)
					for each in example[0]:
					#for each in network.layermatrix[0][0]:
						#print("EACH", each)
						tempinj += (each * num)
						#print("+", tempinj)
				#print("NUM", num)
			#inj = sumi wi,j * ai
			#aj = g(inj)
			#print("J", j)
			#print("1:", network.injmatrix)
			#print("2:", network.ajmatrix)
			#print("TEMP", tempinj)
			#print("LOG", logistic(tempinj))
			network.injmatrix[l][j] = tempinj
			#\print("REG:", tempinj, "LOG", logistic(tempinj))
			network.ajmatrix[l][j] = logistic(tempinj)
			"""
			#print("TEMPPPP", tempinj)
	#print("1:", network.injmatrix)
	#print("2:", network.ajmatrix, example[1])
	#print("*******", network.wmatrix)
	#print("DONE!!!!!!!!")

def backpropagate(network, example, layers):
	nodes = network.nodes
	#c.
	#for l in range(1, len(layers)):
	for j in range(nodes[-1]):
		inj = network.injmatrix[-1][j]
		#print("INJ", inj)
		#print("INJ", inj)
		aj = network.ajmatrix[-1][j]
		#print("AJ", aj)
		#print("AJ", aj)
		#print(logistic(inj))
		#print(example[1])
		network.errorj[-1][j] = logistic(inj)*(1-logistic(inj))*(example[1][j]-aj)
	#print("HERE.", network.errorj)
	#print("ERRORJ:", network.errorj)
	#print("HI", network.errorj)

	#d.
	#print(network.injmatrix)

	for l in range(len(layers) - 2, 0, -1):
		for i in range(layers[l]): #for each node in layer l
			#for n in range(layers[l+1]):
			ini = network.injmatrix[l][i]
			sumj = 0
			for n in range(len(network.wmatrix[l+1][i])):
				sumj += network.wmatrix[l+1][i][n] * network.errorj[l+1][n]
			network.errorj[l][i] = logistic(ini)*(1-logistic(ini)) * sumj
	"""
	for l in range(len(layers) - 2, 0, -1):
		#print("l", l)
		for i in range(layers[l]):
		#	print(i)
			tempsumj = 0
			ini = network.injmatrix[l][i]
			for j in range(layers[l+1]):
				for each in network.wmatrix[l+1][i]:
					tempsumj += each * network.injmatrix[l+1][j]
			network.injmatrix[l][i] = logistic(ini) * (1 - logistic(ini)) * tempsumj
	#print(network.injmatrix)
	"""
	#e.
	#print(network.ajmatrix)

	#print("LOOOOP")
	for l in range(1, len(network.wmatrix)):
		for r in range(len(network.wmatrix[l])):
			#print("--", network.wmatrix[l])
			for n in range(len(network.wmatrix[l][r])):
				#print(l, r, n)
				#print("WMATRix", network.wmatrix[l][r][n])
				#print("ajmatrix", network.ajmatrix[l-1][r])
				#print("INJ", network.injmatrix[l][n])
				network.wmatrix[l][r][n] = network.wmatrix[l][r][n] + 0.1 * network.ajmatrix[l-1][r] * network.errorj[l][n]
				#print(network.injmatrix[l-1][r])
	"""
	for i in range(1, len(network.layers)):
		for j in range(len(network.layers[i])):
			print("LENGTH", len(network.wmatrix[i]))
			for n in range(len(network.wmatrix[i][j])):
				print("i:", i, "j:",  j, "n:", n)
				network.wmatrix[i][j][n] = network.wmatrix[i][j][n] + 0.1 * network.ajmatrix[i-1][j] * network.injmatrix[i][j]
	"""

	"""
	for i in range(1, len(network.wmatrix)):
		for j in range(len(network.wmatrix[i])):
			#print("IJ", i, j)
			#print("----", network.wmatrix[i][j])
			#print("LEN", len(network.wmatrix[i]), len(network.ajmatrix[i]))
			for n in range(len(network.wmatrix[i][j])):
				print(network.wmatrix)
				print()
				print("NETWORK", network.wmatrix[i][j][n], len(network.wmatrix[i][j]))
				print(i, j, n)
			#	print("1", n, network.wmatrix[i][j][n])
			#	print(len(network.ajmatrix[i]), i, j)
			#	print("2", network.ajmatrix[i][j])
			#	print("3", network.injmatrix[i][j])
				network.wmatrix[i][j][n] = network.wmatrix[i][j][n] + 0.1 * network.ajmatrix[i-1][j] * network.injmatrix[i][j]
			#network.injmatrix[l][i] =
	"""

	"""
	for l in range(len(layers) - 1, 1, -1):
		print("len", len(layers))
		for i in range(l):
			print("l", l, "j", i)
			errori = network.errorj[l][i]
			tempinj = 0
			print(".", len(network.errorj))
			for j in network.errorj[l+1]: #l+1[i]
				for wij in network.wmatrix[i]: 
					tempinj += wij * j
					print("???", tempinj)
			#network.errorj = 0
	"""
#"""
#for a in network.wmatrix[l][j]:
#	print("a", a)
#	for inp in network.layermatrix[0][0]:
#		print("inp", inp)
#		tempinj += a*inp
#		print("WHAT", tempinj)
#print("JJJJJ", j)
#network.injmatrix[0][j] = tempinj
#"""
##inj = sumi wi,j*ai
##aj = g()
#		for out in range(layers[-1]):
			#delt[j] = g(inj)/aj (1-g(inj))/aj (Yj - aj)
#		for l in range(len(layers)-1, 0, -1):
#			for j in range(layers[l]):
				#delt[i] = g(inj)(1-g(inj)) sumj (Wi,j*delt[j])

		#input layer






"""
for i in range(1, layers):
	layer = []
	for j in range(nodes[i - 1]):
		row = []
		for k in range(nodes[i]):
			row.append(random.randrange(0, 1))
		layer.append(row)
	self.matrix.append(layer)
"""


"""
class NeuralNetwork:
	def __init__(self, attributes, target):
		self.a0 = attributes[0]
		self.a1 = attributes[1]
		self.a2 = attributes[2]
		self.target = target
		self.w0 = 1.0
		self.w1 = random.randrange(1)
		self.w2 = random.randrange(1)
class InputNode(NeuralNetwork):
	def __init__(self):
		pass
def backpropagate(network):
	pass
	#pass how many layers, how many nodes per layer
	#make a matrix with random weights (represents wi, j)
"""

def crossvalidate(data):
	nums = 1
	for i in range(2, len(data)):
		if len(data) % i == 0:
			nums == 1


def main():
    header, data = read_data(sys.argv[1], ",")

    pairs = convert_data_to_pairs(data, header)

    # Note: add 1.0 to the front of each x vector to account for the dummy input
    training = [([1.0] + x, y) for (x, y) in pairs]
    # Check out the data:
    #for example in training:
        #print(example)
        #inp = NeuralNetwork(example[0], example[1])
        #inputs.append(inp)
    #print(inputs)
    inputs = len(training[0][0])
    #print("LENGTH", inputs)
    net = NeuralNetwork([inputs, 4, 1])


    #print(chunks(training, 5))
    #print(net.wmatrix)
    #for i in range(10000):
    	#for example in training:
    print("***", net.wmatrix)
    #for i in range(100):
    #	for example in training:
    #for i in range(100):
    #for i in range(100):
    #	for example in training:
    for i in range(100):
        for example in training:
            forwardpropagate(net, example, [inputs, 4, 1])
            backpropagate(net, example, [inputs, 4, 1])
    print(accuracy(net, training))
    """
    #For 3 bit incrementer
    for i in range(10000):
       for example in training:
           forwardpropagate(net, example, [inputs, 6, 3])
           backpropagate(net, example, [inputs, 6, 3])
    for n in range(len(training)):
        forwardpropagate(net, training[n], [inputs, 6, 3])
        backpropagate(net, training[n], [inputs, 6, 3])
        print("-----", n, "-----", training[n][1], net.ajmatrix[-1])
    """

    ### I expect the running of your program will work something like this;
    ### this is not mandatory and you could have something else below entirely.
    # nn = NeuralNetwork([3, 6, 3])
    # nn.back_propagation_learning(training)

if __name__ == "__main__":
    main()