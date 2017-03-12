# Generative Adversarial Networks using 1-D input signal

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from matplotlib import style
from scipy.stats import norm

style.use('ggplot')

# Input Data Sample (P_data)
mu, sigma = -1,1
xs=np.linspace(-5,5,1000)
# plt.plot(xs, norm.pdf(xs,loc=mu,scale=sigma))

numTrainIters = 10000
M = 200

# ------------------------------ Multi Layer Perceptron -------------------------
# MultiLayer Perceptron Layers Weights and Bias Value Initialization
# 4-layer ANN with 1 Hidden Layer (6 neurons) and 1 Deeply Connected Layer (5 neurons)
# Total Number of Neurons = 11
# Input => Hidden Layer => Deeply Connected Layer => Output
# Weights are between the layers.
# Biases are on the neurons of the Hidden Layers.
def MultiLayerPerceptron(input, output_dim):
    # Initialize the weights with "empty"
    # Initialize the biases with "0.0"
    init_const = tf.constant_initializer(0.0)
    init_normal = tf.random_normal_initializer()
    # Format: tf.get_variable(name, shape, datatype, initializer, regularizer...)
    # Weights from Input layer to Hidden Layer => W1
    # W1 matrix is of the form: [Input data samples, number of neurons in Hidden Layer]
    w1 = tf.get_variable('w1', [input.get_shape()[1], 6], initializer=init_normal)

    # Bias is added on the Hidden Layer Neuron.
    # Dimension of Bias Matrix: [number of neurons in hidden layer]
    b1 = tf.get_variable('b1', [6], initializer=init_const)

    # Deeply Connected Layer / Second Hidden Layer
    # Weight Dimension: [num neurons in 1st Hidden Layer, num neurons in 2nd Hidden Layer]
    w2 = tf.get_variable('w2', [6, 5], initializer=init_normal)

    # Bias is added on the Hidden Layer Neuron.
    # Dimension of Bias Matrix: [number of neurons in 2nd Hidden Layer]
    b2 = tf.get_variable('b2', [5], initializer=init_const)

    # Weights from Deeply Connected Layer to Output Neuron / Layer
    # W3 matrix has dimensions: [num neurons in hiddenlayer - 1, num neurons in output layer]
    w3 = tf.get_variable('w3', [5, output_dim], initializer=init_normal)

    # Bias is added on the Output Layer.
    # Dimension of Bias Matrix: [number of neurons in Output Layer]
    b3 = tf.get_variable('b3', [output_dim], initializer=init_const)

    # Activation Function (tanh()) for 1st Hidden Layer
    a1 = tf.nn.tanh(tf.matmul(input, w1) + b1)

    # Activation Function (tanh()) for 2nd Hidden Layer
    a2 = tf.nn.tanh(tf.matmul(a1, w2) + b2)

    # Activation Function (tanh()) for Output Layer
    y_Hat = tf.nn.tanh(tf.matmul(a2, w3) + b3)
    return y_Hat, [w1, b1, w2, b2, w3, b3]



# --------------------------- Train the Neural Network ---------------------------
# Using Momentum Optimizer to decrease the Learning Rate as
# we progress so as to reach the minima and not miss it.
# Useful in case when the input is not a Convex function.
def momentumOptimizer(loss, var_list):
    # Initialize the learning rate to a reasonable value
    baseLearningRate = 0.001
    # Set the decay Rate
    decayRate = 0.95
    # Set the step size for decay
    decaySteps = numTrainIters // 4
    batch = tf.Variable(0)
    learning_rate = tf.train.exponential_decay(
        baseLearningRate,
        batch,
        decaySteps,
        decayRate,
        staircase=True
    )
    optimizer = tf.train.MomentumOptimizer(learning_rate, 0.6).minimize(
        loss,
        global_step=batch,
        var_list=var_list
    )
    return optimizer


# ------------------------ Generative Adversarial Network --------------------------
# Pre-Train the Discriminator.
# Doing this saves time at a later stage and only tuning of Discriminator is required.
with tf.variable_scope("D_pre"):
    # Input Data to the Discriminator
    # N sample values in matrix form
    input_node = tf.placeholder(tf.float32, shape=(M, 1))

    # Training Labels = Number of input samples
    train_labels = tf.placeholder(tf.float32, shape=(M, 1))

    # Input the data to Neural Network and get the Output
    # Discriminator sample (D) = Output (y_Hat)
    # Neural Network[input_node, hidden_dim = 6, output_dim = 1]
    D, theta = MultiLayerPerceptron(input_node, 1)

    # Calculate the loss using Mean Squared Error (MSE)
    loss = tf.reduce_mean(tf.square(D - train_labels))

# Optimize the Discriminator for Minimizing Loss / Cost Function.
optimizer = momentumOptimizer(loss, None)



sess = tf.InteractiveSession()
tf.global_variables_initializer().run()


# ------------------------ Plot Initial Decision Surface -----------------------------
def plot_d0(D, input_node):
    f, ax = plt.subplots(1)
    # p_data
    xs = np.linspace(-5, 5, 1000)
    ax.plot(xs, norm.pdf(xs, loc=mu, scale=sigma), label='p_data')
    # decision boundary
    r = 1000  # resolution (number of points)
    xs = np.linspace(-5, 5, r)
    ds = np.zeros((r, 1))  # decision surface
    # process multiple points in parallel in a minibatch
    for i in range(int(r/M)):
        x = np.reshape(xs[M*i:M*(i + 1)],(M,1))
        ds[M*i:M*(i + 1)] = sess.run(D, {input_node: x})

    ax.plot(xs, ds, label='decision boundary')
    ax.set_ylim(0, 1.1)
    plt.legend()

# plot_d0(D, input_node)
# plt.title('Initial Decision Boundary')
# plt.show()
# sess.close()


# ------------------------------------ Plot Training Loss -----------------------------
lh=np.zeros(1000)
for i in range(1000):
    #d=np.random.normal(mu,sigma,M)
    d=(np.random.random(M)-0.5) * 10.0 # instead of sampling only from gaussian, want the domain to be covered as uniformly as possible
    labels=norm.pdf(d,loc=mu,scale=sigma)
    lh[i],_=sess.run([loss,optimizer], {input_node: np.reshape(d,(M,1)), train_labels: np.reshape(labels,(M,1))})

# plt.plot(lh)
# plt.title('Training Loss')
# plt.show()

# plot_d0(D,input_node)

# Theta gets the weights and biases with lowest cost function for Pre Trained Network D.
# Save it.
learnedWeights = sess.run(theta)
sess.close()


# Generator Network
with tf.variable_scope("G"):
    # M: Input Samples generated (Noise Signal)
    z_node=tf.placeholder(tf.float32, shape=(M,1))
    # Get the output and weights of the Generator using Neural Network
    G,theta_g = MultiLayerPerceptron(z_node,1)
    # Scale by 5 to match with range
    G=tf.multiply(5.0,G)


# Discriminator Network:
# D1 => actual Data input Discriminator
# D2 => Output of Generator is Input to Discriminator
with tf.variable_scope("D") as scope:
    # D1 => Trained on Input Data
    x_node = tf.placeholder(tf.float32, shape=(M,1))
    fc,theta_d = MultiLayerPerceptron(x_node,1)
    D1=tf.maximum(tf.minimum(fc,.99), 0.01)
    # make a copy of D that uses the same variables, but takes in G as input
    scope.reuse_variables()

    # D2 => Takes output of Generator as Input
    # Output of Generator (G) input to Discriminator (D2)
    fc,theta_d = MultiLayerPerceptron(G,1)
    D2 = tf.maximum(tf.minimum(fc,.99), 0.01)

# Calculating the Value of Discriminator "D"
# log(D1(x)) + log(1-D2(x))
# We need to Maximize D1 and Minimize value of D2.
# obj_d: Value receieved after processing Input Data
obj_d = tf.reduce_mean(tf.log(D1)+tf.log(1-D2))

# Calculating value of Generator Function "G"
# log(D2(x))
# We need to maximize log(D2(x)) to successfully fool Discriminator
# obj_g: Value received after processing Generator Data
obj_g = tf.reduce_mean(tf.log(D2))

# set up optimizer for G and D
opt_d = momentumOptimizer(1-obj_d, theta_d)
opt_g = momentumOptimizer(1-obj_g, theta_g)



sess=tf.InteractiveSession()
tf.global_variables_initializer().run()



# Copy the Weights saved in Pre-Training over to New D Network
for i,v in enumerate(theta_d):
    sess.run(v.assign(learnedWeights[i]))



def plot_fig():
    # plots pg, pdata, decision boundary
    f,ax=plt.subplots(1)
    # p_data
    xs=np.linspace(-5,5,1000)
    ax.plot(xs, norm.pdf(xs,loc=mu,scale=sigma), label='p_data')

    # decision boundary
    r=5000 # resolution (number of points)
    xs=np.linspace(-5,5,r)
    ds=np.zeros((r,1)) # decision surface
    # process multiple points in parallel in same minibatch
    for i in range(int(r/M)):
        x=np.reshape(xs[M*i:M*(i+1)],(M,1))
        ds[M*i:M*(i+1)]=sess.run(D1,{x_node: x})

    ax.plot(xs, ds, label='decision boundary')

    # distribution of inverse-mapped points
    zs=np.linspace(-5,5,r)
    gs=np.zeros((r,1)) # generator function
    for i in range(int(r/M)):
        z=np.reshape(zs[M*i:M*(i+1)],(M,1))
        gs[M*i:M*(i+1)]=sess.run(G,{z_node: z})
    histc, edges = np.histogram(gs, bins = 10)
    ax.plot(np.linspace(-5,5,10), histc/float(r), label='p_g')

    ax.set_ylim(0,1.1)
    plt.legend()

plot_fig()
plt.title('Before Training')
plt.show()



# Algorithm 1 of Goodfellow et al 2014
# Instead of optimizing with one pair (x,z) at a time, we update the gradient
# based on the average of M loss gradients computed for M different (x,z) pairs.
# The stochastic gradient estimated from a minibatch is closer to the true gradient
# across the training data.
k=1
histd, histg= np.zeros(numTrainIters), np.zeros(numTrainIters)
for i in range(numTrainIters):
    for j in range(k):
        x= np.random.normal(mu,sigma,M) # sampled m-batch from p_data
        x.sort()
        # Sample Batch Noise Input of Dimension "M"
        # Input Noise signal "Z" streached from "-1 to 1" to "-5 to 5"
        z= np.linspace(-5.0,5.0,M)+np.random.random(M)*0.01
        histd[i],_ = sess.run([obj_d,opt_d], {x_node: np.reshape(x,(M,1)), z_node: np.reshape(z,(M,1))})
    # Sample Noise Prior Signal Input
    z= np.linspace(-5.0,5.0,M)+np.random.random(M)*0.01
    # Update Generator
    histg[i],_ = sess.run([obj_g,opt_g], {z_node: np.reshape(z,(M,1))})
    if i % (numTrainIters//10) == 0:
        print((float(i)/float(numTrainIters))*100)


plt.plot(range(numTrainIters),histd, label='obj_d')
plt.plot(range(numTrainIters), 1-histg, label='obj_g')
plt.legend()

plot_fig()
plt.show()

sess.close()

# ----------------------------------- EOC ----------------------------------------------
