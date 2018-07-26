import numpy as np
import os
import platform
from six.moves import cPickle as pickle

def load_pickle(f):
	version = platform.python_version_tuple()
	if version[0] == '2':
		return  pickle.load(f)
	elif version[0] == '3':
		return  pickle.load(f, encoding='latin1')
	raise ValueError("invalid python version: {}".format(version))

def load_CIFAR_batch(filename):
	""" load single batch of cifar """
	with open(filename, 'rb') as f:
		datadict = load_pickle(f)
		X = datadict['data']
		Y = datadict['labels']
		X = X.reshape(10000, 3, 32, 32).transpose(0,2,3,1).astype("float")
		Y = np.array(Y)
		return X, Y

def load_CIFAR10(ROOT):
	""" load all of cifar """
	xs = []
	ys = []
	for b in range(1,6):
		f = os.path.join(ROOT, 'data_batch_%d' % (b, ))
		X, Y = load_CIFAR_batch(f)
		xs.append(X)
		ys.append(Y)	
	Xtr = np.concatenate(xs)
	Ytr = np.concatenate(ys)
	del X, Y
	Xte, Yte = load_CIFAR_batch(os.path.join(ROOT, 'test_batch'))
	return Xtr, Ytr, Xte, Yte

class NearestNeighbor(object):
	def __init__(self):
		pass

	def train(self, X, y):
		""" X is N x D where each row is an example. Y is 1-dimension of size N """
		# the nearest neighbor classifier simply remembers all the training data
		self.Xtr = X
		self.ytr = y

	def predict(self, X):
		""" X is N x D where each row is an example we wish to predict label for """
		num_test = X.shape[0]
		# lets make sure that the output type matches the input type
		Ypred = np.zeros(num_test, dtype = self.ytr.dtype)

		# loop over all test rows
		print(num_test)
		for i in range(10000):
			print(i)
			# find the nearest training image to the i'th test image
			# using the L1 distance (sum of absolute value differences)
			distances = np.sum(np.abs(self.Xtr - X[i,:]), axis = 1)
			min_index = np.argmin(distances) # get the index with smallest distance
			Ypred[i] = self.ytr[min_index] # predict the label of the nearest example
		return Ypred

Xtr, Ytr, Xte, Yte = load_CIFAR10('cifar-10-batches-py') # a magic function we provide
# flatten out all images to be one-dimensional
Xtr_rows = Xtr.reshape(Xtr.shape[0], 32 * 32 * 3) # Xtr_rows becomes 50000 x 3072
Xte_rows = Xte.reshape(Xte.shape[0], 32 * 32 * 3) # Xte_rows becomes 10000 x 3072

nn = NearestNeighbor() # create a Nearest Neighbor classifier class
nn.train(Xtr_rows, Ytr) # train the classifier on the training images and labels
Yte_predict = nn.predict(Xte_rows) # predict labels on the test images
# and now print the classification accuracy, which is the average number
# of examples that are correctly predicted (i.e. label matches)
print('accuracy: %f' % ( np.mean(Yte_predict == Yte) ))