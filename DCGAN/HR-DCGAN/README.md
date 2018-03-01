# High Resolution DCGAN [HR-DCGAN] for MNIST

***This repository contains the code for implementation of HR-DCGAN paper using MNIST Dataset.***

# Requirements

**1. Keras**

**2. Matplotlib**

**3. Numpy [+mkl for Windows]**

# Parts of HR-DCGAN

**a) Generator Architecture:**

**b) Discriminator Architecture:**

**c)  HR-DCGAN Architecture:**

# Difference between DCGAN and HR-DCGAN

**1.** HR-DCGAN uses **Selu** Activation layers with **Batch Normalization** as compared to **ReLU** in case of DCGAN in both Generator and Discriminator.

**2.** HR-DCGAN paper suggests that increasing the batch size for the image produces better results as it provides the model with more data to learn from.

# Usage

**1.** Clone this repository using:

```
git clone https://github.com/anujdutt9/DeepLearning.git
```

and go inside the HR-DCGAN directory.

**2.** Go inside the DCGAN-MNIST directory and run the code using:

**a) Jupyter Notebook**

```
jupyter notebook HR-DCGAN-MNIST.ipynb
````

**b) Python Files**

Open this directory in PyCharm or from command prompt and run the file as:

```
python3 main.py
```

or

```
python main.py
```

# Result

***Coming Soon !!!***
