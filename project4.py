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
		""" Creates empty matrices: one for inputs, activations, errors that are a list of lists:
		a list of each layer that contains a list for each node in that layer,
		as well as a matrix of random wi,j weights from 0, 1"""
		self.layers = len(nodes)
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

	#a. Makes the input activations equal to the actual inputs
	for i in range(len(example[0])):
		network.ajmatrix[0][i] = example[0][i]

	#b. Calculates the inputs and activations for each node in each layer beyond 
	#the first
	for l in range(1, len(layers)):
		for j in range(layers[l]):
			tempinj = 0
			for n in range(len(network.wmatrix[l])):
				tempinj += network.ajmatrix[l-1][n] * network.wmatrix[l][n][j]
			network.injmatrix[l][j] = tempinj
			network.ajmatrix[l][j] = logistic(tempinj)

def backpropagate(network, example, layers):
	""" Propagates backwards to compute the error values and update the weights. """
	nodes = network.nodes
	
	#c. Calculates error for each node in output layer
	for j in range(nodes[-1]):
		inj = network.injmatrix[-1][j]
		aj = network.ajmatrix[-1][j]
		network.errorj[-1][j] = logistic(inj)*(1-logistic(inj))*(example[1][j]-aj)

	#d. Calculates error for each node in the layers before the output layer,
	# working backwards
	for l in range(len(layers) - 2, 0, -1):
		for i in range(layers[l]): #for each node in layer l
			ini = network.injmatrix[l][i]
			sumj = 0
			for n in range(len(network.wmatrix[l+1][i])):
				sumj += network.wmatrix[l+1][i][n] * network.errorj[l+1][n]
			network.errorj[l][i] = logistic(ini)*(1-logistic(ini)) * sumj

	#e. Updates the weight for each wi,j in the weight matrix
	for l in range(1, len(network.wmatrix)):
		for r in range(len(network.wmatrix[l])):
			for n in range(len(network.wmatrix[l][r])):
				network.wmatrix[l][r][n] = network.wmatrix[l][r][n] + 0.1 * network.ajmatrix[l-1][r] * network.errorj[l][n]

"""
def crossvalidate(data):
	nums = 1
	for i in range(2, len(data)):
		if len(data) % i == 0:
			nums == 1
"""

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
    net = NeuralNetwork([inputs, 4, 1])

    #For data like generated.csv and breast-cancer...
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