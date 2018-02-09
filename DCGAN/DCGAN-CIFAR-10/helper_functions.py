# Import Dependencies
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import style
style.use('ggplot')



# Function to Plot Random Sample Images from MNIST Dataset
def plot_sample_images(training_image_data, training_data_size, training_labels):
    # Visualize Images
    fig, ax = plt.subplots(nrows=3, ncols=3, figsize=(10, 5))
    for i in range(0, 3):
        for j in range(0, 3):
            idx = np.random.randint(0, training_data_size, 1)
            idx = idx[0]
            ax[i, j].imshow(training_image_data[idx], cmap='gray')
            ax[i, j].set_axis_off()
            ax[i, j].title.set_text('Label: {}'.format(training_labels[idx]))
            plt.tight_layout()
    plt.show()



# Function to apply Normalization similar to tanh activation function range i.e. [-1,1]
def normalize_images(img):
    img = img.reshape(-1,28,28,1)
    img = np.float32(img)
    img = (img / 255 - 0.5) * 2
    img = np.clip(img, -1, 1)
    return img


# Function to DeNormalize the Images once we are done Training the DCGAN Model
def denormalize_images(img):
    img = (img / 2 + 1) * 255
    img = np.clip(img, 0, 255)
    img = np.uint8(img)
    img = img.reshape(28, 28)
    return img


# Function to Plot Generated Images
def plot_images(generated_images):
    n_images = len(generated_images)
    rows = 4
    cols = n_images // rows

    plt.figure(figsize=(cols, rows))
    for i in range(n_images):
        img = denormalize_images(generated_images[i])
        plt.subplot(rows, cols, i + 1)
        plt.imshow(img, cmap='gray')
        plt.xticks([])
        plt.yticks([])
    plt.tight_layout()
    plt.show()

# ------------------------------- EOC --------------------------------