# Import Dependencies
# Dataset
from keras.datasets.mnist import load_data
from keras.models import Sequential
# Common Layers
from keras.layers import Dense, Activation, BatchNormalization, Reshape, Flatten
from keras.optimizers import Adam
# Layers specific to Generator
from keras.layers import Conv2DTranspose
# Layers specific to Discriminator
from keras.layers import Conv2D, LeakyReLU
# Use this to pass an element-wise TensorFlow/Theano/CNTK function as an activation
import keras.backend as k
# Train Test Split
from sklearn.model_selection import train_test_split
# Import Helper Functions
from helper_functions import *




# Load Dataset
(X_train, y_train), (X_test, y_test) = load_data()

# Get Data Analysis
print('Training Data: \n')
print('Num. Features: ',len(X_train)), print('Num. Labels: ',len(y_train))
print('Shape of Features: ',X_train.shape), print('Shape of Labels: ',y_train.shape)
print('\n\n')

print('Test Data: \n')
print('Num. Features: ',len(X_test)), print('Num. Labels: ',len(y_test))
print('Shape of Features: ',X_test.shape), print('Shape of Labels: ', y_test.shape)

# Shape of One Image
rand_idx = np.random.randint(0, len(X_train), 1)
print('Shape of one Image: ', X_train[rand_idx].shape)


# ----------------------- Visualize Dataset -----------------------------
plot_sample_images(X_train, len(X_train), y_train)


# ----------------------- Data Preprocessing ----------------------------
# Taking a random image and looking at its pixel values
idx = np.random.randint(0, len(X_train), 1)
print('Image Index No.: ', idx)
print('\nImage Pixel Values [Before Normalization]: \n\n',X_train[idx])
print('\n\n Shape of Image: ',X_train[idx].shape)

# Normalize the Training and Test Features
X_train = normalize_images(X_train)
X_test = normalize_images(X_test)

# Test the Normalization Function
sample_img = X_train[3337]
print('Normalized Pixel Values: \n\n', sample_img)
print('\n\n Shape of Normalized Image: ', sample_img.shape)


# ---------------------------- Generator Architecture -------------------------
# Generator
def generator(inputSize):
    generator_model = Sequential()
    # Input Dense Layer
    generator_model.add(Dense(7 * 7 * 128, input_shape=(inputSize,)))
    # Reshape the Input, apply Batch Normalization and Leaky ReLU Activation.
    generator_model.add(Reshape(target_shape=(7, 7, 128)))
    generator_model.add(BatchNormalization())
    generator_model.add(Activation('relu'))

    # First Transpose Convolution Layer
    generator_model.add(Conv2DTranspose(filters=64, kernel_size=5, strides=2, padding='same'))
    generator_model.add(BatchNormalization())
    generator_model.add(Activation('relu'))

    # Since, we are using MNIST Data which has only 1 channel, so filter for Generated Image = 1
    generator_model.add(Conv2DTranspose(filters=1, kernel_size=5, strides=2, padding='same'))
    generator_model.add(Activation('tanh'))

    generator_model.summary()

    return generator_model


# ------------------------------ Discriminator Architecture --------------------
# Discriminator
def discriminator(leakSlope):
    discriminator_model = Sequential()

    # Input and First Conv2D Layer
    discriminator_model.add(Conv2D(filters=64, kernel_size=5, strides=2, padding='same', input_shape=(28, 28, 1)))
    discriminator_model.add(LeakyReLU(alpha=leakSlope))

    # Second Conv2D Layer
    discriminator_model.add(Conv2D(filters=128, kernel_size=5, strides=2, padding='same'))
    discriminator_model.add(BatchNormalization())
    discriminator_model.add(LeakyReLU(alpha=leakSlope))

    # Third Layer
    discriminator_model.add(Flatten())
    discriminator_model.add(Dense(784))
    discriminator_model.add(BatchNormalization())
    discriminator_model.add(LeakyReLU(alpha=leakSlope))

    # Output Layer
    discriminator_model.add(Dense(1))
    discriminator_model.add(Activation('sigmoid'))

    discriminator_model.summary()

    return discriminator_model


# ------------------------------ DCGAN Architecture ----------------------------
# Define DCGAN Architecture
def DCGAN(sample_size, generator_lr, generator_momentum, discriminator_lr, discriminator_momentum, leakyAlpha, show_summary=False):
    # Clear Session
    k.clear_session()

    # Generator
    gen = generator(inputSize=100)

    # Discrimintor
    dis = discriminator(leakSlope=0.2)
    dis.compile(loss='binary_crossentropy', optimizer=Adam(lr=discriminator_lr, beta_1=discriminator_momentum))

    dis.trainable = False

    dcgan = Sequential([gen, dis])
    dcgan.compile(loss='binary_crossentropy', optimizer=Adam(lr=generator_lr, beta_1=generator_momentum))

    if show_summary == True:
        print("\n Generator Model Summary: \n")
        gen.summary()

        print("\n\n Discriminator Model Summary: \n")
        dis.summary()

        print("\n\nDCGAN Model Summary\n")
        dcgan.summary()

    return dcgan, gen, dis



# ---------------------------------------- Train The Model --------------------------
# Function to Train the Model
def train_model(sample_size, generator_lr, generator_momentum, discriminator_lr, discriminator_momentum, leakyAlpha,
                epochs, batch_size, eval_size, smooth):
    # To Do: Add Label Noise Data
    # Training Labels [Real, Fake]
    training_labels = [np.ones([batch_size, 1]), np.zeros([batch_size, 1])]

    # Test Labels [Real, Fake]
    test_labels = [np.ones([eval_size, 1]), np.zeros([eval_size, 1])]

    # Total Number of Batches = (Total Training Images / Images per Batch)
    num_batches = (len(X_train) // batch_size)

    # Call the DCGAN Architecture
    dcgan, generator, discriminator = DCGAN(sample_size, generator_lr, generator_momentum, discriminator_lr,
                                            discriminator_momentum, leakyAlpha, show_summary=False)

    # Array to Store Cost/Loss Values
    cost = []

    # Train the Generator and Discriminator
    for i in range(epochs):
        for j in range(num_batches):
            # Noise Input for Generator
            # Mean = 0, Stddev = 1
            noise_data = np.random.normal(loc=0, scale=1, size=(batch_size, sample_size))

            # Make Predictions using Generator and Generate Fake Images
            fake_images = generator.predict_on_batch(noise_data)

            # Load MNIST Data in Batches
            # [0:128], [128:256], ...
            train_image_batch = X_train[j * batch_size:(j + 1) * batch_size]

            # Train the Discriminator
            discriminator.trainable = True

            # Train the Discriminator on Training Data and Labels
            discriminator.train_on_batch(train_image_batch, training_labels[0] * (1 - smooth))

            # Train Discriminator on Fake Generated Images and Labels
            discriminator.train_on_batch(fake_images, training_labels[1])

            # Set Discriminator training to False when Generator is Training
            discriminator.trainable = False

            # Train the Generator on Noise Data Input with Training Labels to reduce Cost/Loss
            # This way, the Discriminator gets trained twice for each one training step of Generator
            dcgan.train_on_batch(noise_data, training_labels[0])

        # To Do: Add Eval Code
        # Eval/Test Features [Real,Fake]
        real_eval_features = X_test[np.random.choice(len(X_test), size=eval_size, replace=False)]

        # Eval Noise Data
        noise_data = np.random.normal(loc=0, scale=1, size=(eval_size, sample_size))

        # Fake Eval Features: Creates the Images to Fool the Discriminator
        fake_eval_features = generator.predict_on_batch(noise_data)

        # Calculate Loss
        # Discriminator Loss: Actual Training Loss for Classification + Loss on Fake Data
        discriminator_loss = discriminator.test_on_batch(real_eval_features, test_labels[0])
        discriminator_loss += discriminator.test_on_batch(fake_eval_features, test_labels[1])

        # Generator Loss: DCGAN Loss
        generator_loss = dcgan.test_on_batch(noise_data, test_labels[0])

        # Add calculated cost/loss to array for plotting
        cost.append((discriminator_loss, generator_loss))

        print("Epochs: {0}, Generator Loss: {1}, Discriminator Loss: {2}".format(i + 1, generator_loss, discriminator_loss))

        # Plot the Images and Save them after every 20 epochs
        if ((i + 1) % 20 == 0):
            plot_images(fake_eval_features)

    # Save Trained Models
    generator.save('./TrainedModel/mnist_generator.h5')
    discriminator.save('./TrainedModel/mnist_discriminator.h5')
    dcgan.save('./TrainedModel/mnist_dcgan.h5')



# Main Function
if __name__ == '__main()__':
    train_model(sample_size=100,
                generator_lr=0.0001,
                generator_momentum=0.9,
                discriminator_lr=0.001,
                discriminator_momentum=0.9,
                leakyAlpha=0.01,
                epochs=100,
                batch_size=128,
                eval_size=16,
                smooth=0.1)
    print("Model Training Complete !!")

# --------------------------------- EOC -------------------------------