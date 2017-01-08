import numpy as np
import Neural_Network as nn
import trainer as t

# X = (hours sleeping, hours studying), y = Score on test

# Training Data:
# trainX = np.array(([3,5], [5,1], [10,2], [6,1.5]), dtype=float)
# trainY = np.array(([75], [82], [93], [70]), dtype=float)
trainX = np.array(([10, 10], [15, 8], [5, 19], [2, 200]), dtype=float)
trainY = np.array(([100], [120], [95], [400]), dtype=float)

# Testing Data:
# testX = np.array(([4, 5.5], [4.5,1], [9,2.5], [6, 2], [10,3]), dtype=float)
# testY = np.array(([70], [89], [85], [75], [92]), dtype=float)
testX = np.array(([1, 5], [6, 12], [7, 7], [10, 5]), dtype=float)
testY = np.array(([5], [72], [49], [50]), dtype=float)

# Normalize:
# trainX = trainX/np.amax(trainX, axis=0)
# trainY = trainY/100 #Max test score is 100

# Normalize by max of training data:
# testX = testX/np.amax(trainX, axis=0)
# testY = testY/100 #Max test score is 100

NN = nn.Neural_Network()
T = t.trainer(NN)
T.train(trainX, trainY, testX, testY)

print testX
print testY
print NN.forward(testX)
