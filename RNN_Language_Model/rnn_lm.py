# Recurrent Neural Network based Language Model
# Aim: To train the NN so that it can generate random text using words from Vocabulary which is
#      grammatically correct and makes some sense.
#      The vocabulary is made using the input raw text data

# NOTE: This code is computationally very expensive. Use it for understanding the underlying concepts only.

# Import Dependencies
import os
import sys
import csv
import itertools
import operator
import numpy as np
import pickle
import nltk
from nltk import sent_tokenize, word_tokenize, FreqDist
from utils import *
import matplotlib.pyplot as plt



directory = 'SavedModels'
if not os.path.exists(directory):
    os.makedirs(directory)

np.random.seed(0)

# Step1: Training Data and Preprocessing
#       a) Tokenize the Input text (sentence to words)
#       b) Form the Vocabulary and remove Infrequent words
#       c) Add "Start" and "End" Tokens to the sentences

# Vocabulary Size: Top 8000 words selected from a list of words according to their frequency of occurence
vocab_size = 8000

# Token to replace the infrequent words
unknown_token = 'Unknown_Token'

# Sentence start and end tokens
sentence_start_token = 'Sentence_Start'
sentence_end_token = 'Sentence_End'

# Read the text file
with open('reddit-comments-2015-08.csv', encoding='utf-8') as f:
    # Read the contents line by line
    reader = csv.reader(f, skipinitialspace = True)
    # Split comments into sentences
    sentences = itertools.chain(*[sent_tokenize(rows[0].lower()) for rows in reader])
    # Append the start and end to sentences
    sentences = ['%s %s %s' % (sentence_start_token, sent, sentence_end_token) for sent in sentences]

# Save the Tokenized Sentences
pickle_out = open('SavedModels/Sentences.pkl', 'wb')
pickle.dump(sentences, pickle_out)
pickle_out.close()

pickle_in = open('SavedModels/Sentences.pkl', 'rb')
sentences = pickle.load(pickle_in)
print('Parsed {} sentences.'.format(len(sentences)))


# ------------ (a) -----------
# Tokenize the input sentences
tokenized_sentences = [word_tokenize(sent) for sent in sentences]
print('Tokenized Sentences: \n', tokenized_sentences)

# Count word frequency to form the Vocabulary
word_freq = FreqDist(itertools.chain(*tokenized_sentences))
print('Found {} frequently used words.'.format(len(word_freq.items())))


# ------------- (b) -------------
# Build the Vocabulary using most frequent words
vocabulary = word_freq.most_common(vocab_size-1)

# Save the Vocabulary as a Pickle File to save loading time of whole data again
pickle_out = open('SavedModels/vocabulary.pkl', 'wb')
pickle.dump(vocabulary, pickle_out)
pickle_out.close()

pickle_in = open('SavedModels/vocabulary.pkl', 'rb')
vocab = pickle.load(pickle_in)
print('Vocabulary Size {}'.format(vocab_size))
print('Least frequent word in the vocab is {} and appeared {} times'.format(vocab[-1][0], vocab[-1][1]))

# Map each word to an Index
# Words corresponding to the Index
print('\nMapping Index to Words...')
index_to_word = [x[0] for x in vocab]
index_to_word.append(unknown_token)

# Prints Word corresponding to Input Index; 203 -> 'different'
print('Test idx2w: The Word corresponding to Index 203 is ',index_to_word[203])

# Word to Index Mapping
print('\nMapping Word to Indexes...')
word_to_index = dict([(w,i) for i,w in enumerate(index_to_word)])

# Prints Index corresponding to Input word; 'different' -> 203
print('Test w2idx: The Index corresponding to Word '+ index_to_word[203]+ ' is ',word_to_index['day'])


# Replace all words not in vocabulary with unknown_token
for i,sent in enumerate(tokenized_sentences):
    tokenized_sentences[i] = [x if x in word_to_index else unknown_token for x in sent]

# Save the Tokenized Sentences as pickle file
pickle_out = open('SavedModels/Tokenized_Sentences.pkl', 'wb')
pickle.dump(tokenized_sentences, pickle_out)
pickle_out.close()

pickle_in = open('SavedModels/Tokenized_Sentences.pkl', 'rb')
tokenized_sentences = pickle.load(pickle_in)
print('\nSentence structure before pre-processing: ', sentences[0])
print('Sentence structure after pre-processing: ',tokenized_sentences[0])


# Create the Training Data
# X_train: Features
# Features are the array of all indices corresponding to all words in the tokenized sentences.
X_train = np.asarray([[word_to_index[word] for word in sent[:-1]] for sent in tokenized_sentences])

# y_train: X_train shifted by one position
# We need to predict the next word. If x_train[1] is 1st word then, y_train[1] is the second word in same sentence.
y_train = np.asarray([[word_to_index[word] for word in sent[1:]] for sent in tokenized_sentences])

x = X_train[1]
y = y_train[1]

print('\nSample Training Data(X_train) [Starts from First word of sentence till Second Last word]:')
for i in range(len(x)):
    print(index_to_word[x[i]],end=' ')

print('\n')

print('\nSample Training Data(y_train) [Starts from Second word of sentence till Last word of Sentence]:')
for i in range(len(y)):
    print(index_to_word[y[i]], end=' ')


# ------------------------------------------------------- Part 2 ------------------------------------------------


class RNN(object):
    # Initialize Variables
    def __init__(self,word_dim, hidden_dim = 100, bptt_truncate = 4):
        # Input Word Dimensions; Size of Vocabulary: 8000
        self.word_dim = word_dim
        # Hidden Layer Dimensions; 100 neurons; 100x8000
        self.hidden_dim = hidden_dim
        self.bptt_truncate = bptt_truncate

        # Randomly Initialize Neural Network Parameters
        # Recommended Values: [-1/sqrt(n), 1/sqrt(n)] for using "tanh" non-linearity
        # np.random.uniform(low, high, size)
        # W1: 100 x 8000;  W2: 8000 x 100
        # W1: Weight Matrix from Input to Hidden Layer
        self.W1 = np.random.uniform(-np.sqrt(1./word_dim), np.sqrt(1./word_dim), (hidden_dim, word_dim))
        # W2:Weight Matrix from Current Hidden Layer to Output
        self.W2 = np.random.uniform(-np.sqrt(1./word_dim), np.sqrt(1./word_dim), (word_dim, hidden_dim))
        # Wh: Weight Matrix from Previous Hidden Layer to Current Hidden Layer
        self.Wh = np.random.uniform(-np.sqrt(1./word_dim), np.sqrt(1./word_dim), (hidden_dim, hidden_dim))



    # Forward Propagation: Predicting Word Probabilities
    def forwardPropagation(self, X):
        # T: Number of Time steps => Input Matrix Size 8000 x 8000
        T = len(X)
        # s: Saves all hidden states during Forward Propagation
        state_t = np.zeros((T+1, self.hidden_dim))
        # Initial Previous Hidden state is "0"
        state_t[-1] = np.zeros(self.hidden_dim)
        # Save Output at each time step
        out_t = np.zeros((T, self.word_dim))
        for t in np.arange(T):
            # Input "X" is a array of 8000 indices with all "0's" and a index value only where the word at that time step is present
            # We need to convert Input "X" into one hot vector i.e "0's" and "1's".
            # With only single "1" in X and multiplying it with Weight W1, only one row of values of W will remain, rest all "0's"
            # Hence, select only that row at time step "t" from W equivalent to One Hot Product with "W".
            # Also dot product of Previous hidden state with Weight Matrix "Wh"
            state_t[t] = np.tanh(self.W1[:,X[t]] + self.Wh.dot(state_t[t-1]))
            # Output at current time step is dot product of Weights "W2" and current Hidden State Value from above
            # out_t: is the output vector of size 8000 containing probability of each word corresponding to index in vocabulary
            # Shape of Output: (number of words, vocab_size)
            out_t[t] = softmax(self.W2.dot(state_t[t]))
        return [out_t, state_t]



    # We get the output vector of size 8000 with probability of each word
    # If we want only the word with highest probability as the final output, predict function does that
    # For each word in the sentence (in form of indexes "X"), the RNN makes 8000 predictions
    # Choose the best predictions for each word and that forms the sentence
    def predict(self, X):
        # Perform Forward Propagations and return Index of the Highest Probability in Output vector at each time step
        # Shape of Output: (number of words, vocab_size)
        out, state = self.forwardPropagation(X)
        # Return the max probability value from the output vector
        return np.argmax(out, axis=1)



    # Cross Entropy Loss Function
    def total_Loss(self, X, y):
        loss = 0
        # For each word in "y_train", calculate the output and the state
        for i in np.arange(len(y)):
            out, state = self.forwardPropagation(X[i])
            # Correct Word Predictions
            # Outputs a probability vector for each word in the input vector
            correct_word_predictions = out[np.arange(len(y[i])), y[i]]
            # Add to loss based on how far away we were from actual index value
            loss += -1 * np.sum(np.log(correct_word_predictions))
        return loss


    # Divide the Calculated Loss by "N"
    def calculateLoss(self, X, y):
        # Divide the total loss by "N"
        # y is an array of arrays, so access it using y_i
        # N: length of array containing words; number of input words
        N = np.sum((len(y_i) for y_i in y))
        return self.total_Loss(X,y)/N




    # Back Propagation Through Time (BPTT)
    # Now we need to back propagate the error i.e dL/dW1, dL/dW2, dL/dWh.
    # Also since, dL/dWh is from a previous time step, we need to start Backpropagation from the
    # last word predicted to the first word prediction and go on till the LOSS is reduced.
    def BPTT(self, X, y):
        # Define the timesteps
        # Timesteps will be equal to the number of outputs, i.e size of "y"
        T = len(y)
        # Propagate Forward to calculate the output and hence, the "Loss" at each timestep
        out, state = self.forwardPropagation(X)
        # Define placeholders for Derivatives / Gradients
        # dL/dW1, dL/dW2, dL/dWh
        dLdW1 = np.zeros(self.W1.shape)
        dLdW2 = np.zeros(self.W2.shape)
        dLdWh = np.zeros(self.Wh.shape)
        error_out = out
        error_out[np.arange(len(y)), y] -= 1
        # For each output, Back Propagate starting from the last word/output predicted
        for t in np.arange(T)[::-1]:
            # Compute the derivative at time step "t"
            dLdW2 += np.outer(error_out[t], state[t].T)
            # Calcuate Initial error at time step "t"
            error_t = self.W2.T.dot(error_out[t])*(1-(state[t]**2))

            # Back Propagation through time for "bptt" steps; "4" steps here
            for bptt_step in np.arange(max(0, t-self.bptt_truncate), t+1)[::-1]:
                # print('Backpropagation step t = {} , bptt step = {}'.format(t,bptt_step))
                dLdWh += np.outer(error_t, state[bptt_step-1])
                dLdW1[:,X[bptt_step]] += error_t
                # Update delta error for next step
                error_t = self.Wh.T.dot(error_t)*(1 - state[bptt_step-1]**2)
            return [dLdW1, dLdW2, dLdWh]


    # Gradient Checking to see that the Backpropagation is working correctly
    def gradient_check(self, X, y, h=0.001, error_threshold=0.01):
        # Calculate the gradients using Backpropagation
        bptt_gradients = self.BPTT(X, y)
        # List of all parameters we want to check
        model_parameters = ['W1', 'W2', 'Wh']
        # Gradient check for each parameter
        for pidx, pname in enumerate(model_parameters):
            # Get the actual parameter value using name
            parameter = operator.attrgetter(pname)(self)
            # Iterate over each element of the parameter matrix
            # ex. (0,0), (0,1) ....
            # W1: 100x8000, W2: 8000x100, Wh: 100x100
            it = np.nditer(parameter, flags = ['multi_index'], op_flags = ['readwrite'])
            while not it.finished:
                ix = it.multi_index
                # Save original value so we can reset it later
                original_value = parameter[ix]
                # Estimate the gradient using (f(x+h) - f(x-h))/(2*h)
                parameter[ix] = original_value+h
                gradplus = self.calculateLoss([X],[y])
                parameter[ix] = original_value-h
                gradminus = self.calculateLoss([X],[y])

                estimated_gradient = (gradplus - gradminus)/(2*h)
                # Reset Parameter to Original Value
                parameter[ix] = original_value
                # Calculate Gradient for this Parameter using Back Propagation
                backprop_gradient = bptt_gradients[pidx][ix]
                # Calculate Relative Error: (|X - y|/(|X| + |y|))
                relative_error = np.abs(backprop_gradient-estimated_gradient)/(np.abs(backprop_gradient)+np.abs(estimated_gradient))
                # If the error is too large, gradient check is failed
                if relative_error > error_threshold:
                    print('Gradient Check ERROR: parameter = {}, ix = {}'.format(pname, ix))
                    print('+h Loss: {}'.format(gradplus))
                    print('-h Loss: {}'.format(gradminus))
                    print('Estimated Gradient: {}'.format(estimated_gradient))
                    print('Backpropagation Gradient: {}'.format(backprop_gradient))
                    print('Relative Error: {}'.format(relative_error))
                    return
                it.iternext()
            print('Gradient Check for Parameter {} Passed.'.format(pname))


    # Perform Stochastic Gradient Descent (SGD)
    def SGD_step(self, X, y, learning_rate):
        # Calculate the Gradients
        dLdW1, dLdW2, dLdWh = self.BPTT(X, y)
        # Update parameters (Weights) according to Gradients and Learning Rate
        self.W1 -= learning_rate * dLdW1
        self.W2 -= learning_rate * dLdW2
        self.Wh -= learning_rate * dLdWh



    # Outer SGD Loop
    # model: RNN Model
    # X_train: Training Dataset
    # y_train: Training data labels
    # learning_rate: Initial Learning Rate for SGD
    # n_epochs: Number of times to iterate through the complete data
    # evaluate_loss_steps: Evaluate the loss after these many epochs
    def SGD_Train(self, model, X_train, y_train, learning_rate=0.005, n_epochs=100, evaluate_loss_steps=5):
        # Keep Track of Losses
        losses = []
        num_examples_seen = 0
        for epoch in range(n_epochs):
            if(epoch % evaluate_loss_steps == 0):
                loss = model.calculateLoss(X_train, y_train)
                losses.append((num_examples_seen, loss))
                print('Loss after num_examples_seen = {}, epoch = {}, {}'.format(num_examples_seen, epoch, loss))
                # Adjust the Learning Rate if loss Increases
                if((len(losses)) > 1 and (losses[-1][1] > losses[-2][1])):
                    learning_rate *= 0.5
                    print('Setting Learning Rate to {}'.format(learning_rate))
                sys.stdout.flush()

            # For each Training Example
            for i in range(len(y_train)):
                # One SGD Step
                model.SGD_step(X_train[i], y_train[i], learning_rate)
                num_examples_seen += 1



# Generate Sentences from the Trained Model
def generate_sentences(model):
    # Start the sentences with start token
    new_sentence = [word_to_index[sentence_start_token]]
    #Repeat until we get stop token
    while not new_sentence[-1] == word_to_index[sentence_end_token]:
        next_word_probb,_ = model.forwardPropagation(new_sentence)
        sampled_word = word_to_index[unknown_token]
        #We don't want infrequent words
        while sampled_word == word_to_index[unknown_token]:
            samples = np.random.multinomial(1, next_word_probb[-1])
            sampled_word = np.argmax(samples)
        new_sentence.append(sampled_word)
    sentence_str = [index_to_word[x] for x in new_sentence[1:-1]]
    return sentence_str





# ----------------------------- Testing --------------------------------
if __name__ == '__main__':
    print('\nLoading RNN Model...\n')
    model = RNN(word_dim=vocab_size)
    # Uncomment to test the model as Loss Decreases over time
    # losses = model.SGD_Train(model, X_train[:100], y_train[:100], n_epochs=10, evaluate_loss_steps=1)

    print('\nSGD Step called...\n')
    model.SGD_step(X_train[10], y_train[10], learning_rate=0.005)
    print('\nTraining SGD...\n')
    model.SGD_Train(model,X_train,y_train)

    print('\nSaving Trained Model...\n')
    pickle_out = open('SavedModels/TrainedModel.pkl', 'wb')
    pickle.dump(model, pickle_out)
    pickle_out.close()
    pickle_in = open('SavedModels/TrainedModel.pkl', 'rb')
    model = pickle.load(pickle_in)

    print('\nSentence Predictions...\n')
    num_sentences = 10
    senten_min_length = 7

    for i in range(num_sentences):
        sent = []
        # We want long sentences, not sentences with one or two words
        while len(sent) < senten_min_length:
            sent = generate_sentences(model)
        print(" ".join(sent))

# ------------------------------- EOC -----------------------------------