# CNN Layer Activation Visualization

# Import Dependencies
import math
import numpy as np
import tensorflow as tf
import tensorflow.contrib.slim as slim
from tensorflow.examples.tutorials.mnist import input_data
import matplotlib.pyplot as plt
from matplotlib import style
style.use('ggplot')




# Define the Layers of Convolutional Neural Network
def convolutionalNeuralNetwork():
    # First Set of Input -> Convolution -> Pooling Layer
    # Input
    X_img = tf.reshape(X,shape=[-1,28,28,1])
    # First Hidden Layer
    hidden_1 = slim.conv2d(X_img,5,kernel_size=[5,5])
    # Pooling Layer
    pool_1 = slim.max_pool2d(hidden_1,kernel_size=[2,2])

    # Second Set of Input -> Convolution -> Pooling Layer
    # Input = Previous Pooling Layer (pool_1)
    # Second Hidden Layer
    hidden_2 = slim.conv2d(pool_1,num_outputs=5,kernel_size=[5,5])
    # Pooling Layer
    pool_2 = slim.max_pool2d(hidden_2,kernel_size=[2,2])

    # Third Set of Input -> Convolution -> Pooling Layer
    # Input = Previous Pooling Layer (pool_2)
    # Third Hidden Layer
    hidden_3 = slim.conv2d(pool_2,num_outputs=25,kernel_size=[5,5])
    # Pooling Layer
    pool_3 = slim.dropout(hidden_3,probb)

    # Last Layer of CNN: Fully Connected Layer
    output_layer = slim.fully_connected(slim.flatten(hidden_3),num_outputs=10,activation_fn=tf.nn.softmax)

    return hidden_1,hidden_2,hidden_3,output_layer



# Get Activations from Hidden Layers and Plot Them
def plotCNNFilter(layer,stimuli):
    # order: 'F’ means to read / write the elements using Fortran-like index order,
    # with the first index changing fastest, and the last index changing slowest.
    layers = sess.run(layer, feed_dict={X: np.reshape(stimuli, [1, 784], order='F'), probb: 1.0})
    filters = layers.shape[3]
    plt.figure(1, figsize=(20,20))
    n_columns = 6
    n_rows = math.ceil(filters / n_columns) + 1
    for i in range(filters):
        plt.subplot(n_rows, n_columns, i+1)
        plt.title('Filter: {0} '.format(str(i)))
        plt.imshow(layers[0,:,:,i], interpolation="nearest")
    plt.show()



# Main Function
if __name__ == '__main__':

    # Import Data
    data = input_data.read_data_sets('./Dataset/', one_hot=True)

    # Reset default Graph
    tf.reset_default_graph()

    # Define Features and Labels
    X = tf.placeholder(tf.float32, shape=[None, 784], name='Feature')
    y = tf.placeholder(tf.float32, shape=[None, 10], name='Label')
    probb = tf.placeholder('float')

    hidden_1, hidden_2, hidden_3, output_layer = convolutionalNeuralNetwork()

    # Cross Entropy
    cross_entropy = -tf.reduce_sum(y * tf.log(output_layer))

    # Predicted Labels
    y_pred = tf.equal(tf.arg_max(input=output_layer, dimension=1), tf.arg_max(input=y, dimension=1))

    # Accuracy
    acc = tf.reduce_mean(input_tensor=tf.cast(x=y_pred, dtype='float'))

    # Optimizer / Training Step
    training_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
    training_step2 = tf.train.GradientDescentOptimizer(1e-4).minimize(cross_entropy)

    batch_size = 50
    sess = tf.Session()
    init = tf.global_variables_initializer()
    sess.run(init)

    for i in range(1001):
        batch = data.train.next_batch(batch_size)
        sess.run(training_step,feed_dict={X:batch[0],y:batch[1],probb:0.5})

        if (i%100 == 0) and (i != 0):
            trainAccuracy = sess.run(acc, feed_dict={X:batch[0],y:batch[1],probb:1.0})
            print('Step: {0}, Training Accuracy: {1}'.format(i,trainAccuracy))


    testAccuracy = sess.run(acc, feed_dict={X: data.test.images, y: data.test.labels, probb: 1.0})
    print('Test Accuracy: ',testAccuracy)

    # Plot the Input Test Image
    n = np.random.randint(0,9,1)
    imageToUse = data.test.images[n]
    plt.imshow(np.reshape(imageToUse, [28, 28]), interpolation="nearest", cmap="gray")
    plt.title('Input Test Image')
    plt.show()


    # Plot the Activations at each CNN Layer
    layers = [hidden_1,hidden_2,hidden_3]

    # Plot out the Layers
    for layer in layers:
        plotCNNFilter(layer,imageToUse)

# ------------------------ EOC -----------------------------