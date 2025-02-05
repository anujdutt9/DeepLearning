# Basic 3 Layer Feed Forward Neural Network with Back Propagation from Scratch

import numpy as np
from scipy import optimize
import matplotlib.pyplot as plt

# Sample Input Data
x = np.array(([3,5],[5,1],[10,2]), dtype=float)
y = np.array(([75],[82],[93]),dtype=float)

# Normalizing the Data
X = x / np.amax(x,axis=0)
y = y / 100

# -------------------- 3-Layer Feed Forward Neural Network with Back Propagation ----------------------
class NeuralNetwork(object):
    # Define Hyperparameters
    # Hyperparameters are constant and define behaviour of NN
    def __init__(self):
        self.inputLayerSize = 2
        self.outputLayerSize = 1
        self.hiddenLayerSize = 3
        # Initialize Weights (Parameters)
        # W(1): All weights from Input layer to Hidden Layer
        # W(2): All weights from Hidden Layer to Output Layer

        # np.random.randn(x,y,...): Returns random values in a given shape
        # Returns a matrix of form 2x4
        # Values in 1st row: W11, W12, W13, W14;  => W(1)
        # Values in second row: W21, W22, W23, W24;  => W(1)
        self.W1 = np.random.randn(self.inputLayerSize,self.hiddenLayerSize)

        # Returns a matrix of random values in shape 4x1 matrix
        # 4 rows, 1 column
        # Values: W11, W21, W31, W41 => W(2)
        self.W2 = np.random.randn(self.hiddenLayerSize,self.outputLayerSize)


    # Propogate Inputs through Network
    def forward(self, X):
        # Pass in matrices instead of individual values
        # leading to Speedup.

        # Input to Hidden Layer Z(2) values
        self.z2 = np.dot(X, self.W1)
        # print('Z2:\n',self.z2)

        # Using Sigmoid Activation Function to convert values in range (0,1)
        self.a2 = self.sigmoid(self.z2)
        # print('a2:\n', self.a2)

        # Hidden Layer to Output Layer Z(3) values
        self.z3 = np.dot(self.a2,self.W2)
        # print('Z3:\n', self.z3)

        # Final Activation Function gives the Output
        y_Hat = self.sigmoid(self.z3)
        return y_Hat


    # Activation Function
    # Applied to each element of matrix "Z = X.W"
    def sigmoid(self, z):
        return 1 / (1 + np.exp(-z))


    # Function to return Prime (Diffrentiation) of f(Z) wrt Z
    def sigmoidPrime(self,z):
        return np.exp(-z)/((1+np.exp(-z))**2)


    # Calculate the Cost i.e error between predicted and actual values
    def costFunction(self, X,y):
        self.y_Hat = self.forward(X)
        J = 0.5*sum((y-self.y_Hat)**2)
        return J


    # Calculation of BackPropagation error
    # This helps to update the weights and tune them to reducse cost function
    def costFunctionPrime(self, X, y):
        self.y_Hat = self.forward(X)

        delta3 = np.multiply(-(y-self.y_Hat), self.sigmoidPrime(self.z3))
        dJdW2 = np.dot(self.a2.T, delta3)

        delta2 = np.dot(delta3, self.W2.T)*self.sigmoidPrime(self.z2)
        dJdW1 = np.dot(X.T, delta2)
        return dJdW1, dJdW2



# ---------------- Testing the Gradient Computation Part (costFunctionPrime) ----------------------
#     Test if the values of dJdW1 and dJdW2 obtained are accurate or not.

    # Helper Functions for interacting with other classes:
    def getParams(self):
        # Get W1 and W2 unrolled into vector:
        params = np.concatenate((self.W1.ravel(), self.W2.ravel()))
        return params

    def setParams(self, params):
        # Set W1 and W2 using single paramater vector.
        W1_start = 0
        W1_end = self.hiddenLayerSize * self.inputLayerSize
        self.W1 = np.reshape(params[W1_start:W1_end], (self.inputLayerSize, self.hiddenLayerSize))
        W2_end = W1_end + self.hiddenLayerSize * self.outputLayerSize
        self.W2 = np.reshape(params[W1_end:W2_end], (self.hiddenLayerSize, self.outputLayerSize))

    # Compute the gradient Values for each weight in W1 and W2.
    def computeGradients(self, X, y):
        dJdW1, dJdW2 = self.costFunctionPrime(X, y)
        return np.concatenate((dJdW1.ravel(), dJdW2.ravel()))


# Finding Slope of x^2(Weights)
# Computing slope for all values of Weights

# If the Weight matrix computed by computeGradients matches the
# matrix generated by computing Numerical Gradients, then the NN works fine.
def computeNumericalGradient(N, X, y):
    paramsInitial = N.getParams()
    numgrad = np.zeros(paramsInitial.shape)
    perturb = np.zeros(paramsInitial.shape)
    e = 1e-4

    for p in range(len(paramsInitial)):
        # Set perturbation vector
        perturb[p] = e

        # Computing (x + epsilon): value above test point
        N.setParams(paramsInitial + perturb)
        loss2 = N.costFunction(X, y)

        # Computing (x - epsilon): value below test point
        N.setParams(paramsInitial - perturb)
        loss1 = N.costFunction(X, y)

        # Compute Numerical Gradient; Slope of x^2
        numgrad[p] = (loss2 - loss1) / (2 * e)

        # Return the value we changed to zero:
        perturb[p] = 0

    # Return Params to original value:
    N.setParams(paramsInitial)
    return numgrad


# ------------------------ Training the Neural Network ------------------------------
class trainer(object):
    def __init__(self, N):
        # Make Local reference to network:
        self.N = N

    def callbackF(self, params):
        self.N.setParams(params)
        self.J.append(self.N.costFunction(self.X, self.y))

    def costFunctionWrapper(self, params, X, y):
        self.N.setParams(params)
        cost = self.N.costFunction(X, y)
        grad = self.N.computeGradients(X, y)
        return cost, grad

    def train(self, X, y):
        # Make an internal variable for the callback function:
        self.X = X
        self.y = y

        # Make empty list to store costs:
        self.J = []

        params0 = self.N.getParams()
        options = {'maxiter': 200, 'disp': True}
        _res = optimize.minimize(self.costFunctionWrapper, params0, jac=True, method='BFGS', \
                                     args=(X, y), options=options, callback=self.callbackF)
        self.N.setParams(_res.x)
        self.optimizationResults = _res



# -------------------- Testing the Neural Network --------------------------
nn = NeuralNetwork()
y_Hat = nn.forward(X)
print('Estimated Values (y_Hat): \n',y_Hat)
print('\n Actual Values (y): \n',y)

# Cost Function without Back Propagation
J = nn.costFunction(X,y)
print('Cost1 Function J: \n', J)

# Back propagated derivatives of cost function
dJdW1, dJdW2 = nn.costFunctionPrime(X,y)
print('dJdW1: \n',dJdW1)
print('\ndJdW2: ',dJdW2)

# Adding the derivatives to the Initial weights (tuning weights)
# Use only if the dJ/dW is -ve.
scalar = 3
nn.W1 = nn.W1 + scalar*dJdW1
nn.W2 = nn.W2 + scalar*dJdW2
cost2 = nn.costFunction(X,y)
print('cost2: \n', cost2)


# Subtracting scalar times the Back Propagated loss to tune weights
# Use only if the dJ/dW is +ve.
nn.W1 = nn.W1 - scalar*dJdW1
nn.W2 = nn.W2 - scalar*dJdW2
cost3 = nn.costFunction(X,y)
print('cost3: \n', cost3)


# Checking for the Accuracy of Numerical Gradient
grad = nn.computeGradients(X,y)
print('Original Gradient: \n',grad)

num_grad = computeNumericalGradient(nn,X,y)
print('Numerical Gradient: \n', num_grad)


# Training the NN
T = trainer(nn)
T.train(X,y)
plt.plot(T.J)
plt.show()
print('\n')


# --------------------- Testing Trained Neural Network ---------------

# Check Cost after Training
cost = nn.costFunction(X,y)
print('Cost after Training: \n',cost)
print('\n')

print('Actual value: \n', y)
print('\n')

predicted_val = nn.forward(X)
print('Predicted Values: \n',predicted_val)
print('\n')

# ----------------------------------- EOC ----------------------------------
