# Introduction to Deep Learning WS2019
This repository holds my solutions for the [Introduction to Deep Learning](https://niessner.github.io/I2DL/) course of the winter semester 2019 held by [Prof. Dr. Matthias Nießner](https://www.niessnerlab.org/members/matthias_niessner/profile.html) and [Dr. Angela Dai](https://angeladai.github.io/). The course was offered at [Technische Universität München (TUM)](https://www.tum.de/).

***

## Exercise 0
### Introduction
- Introduction to IPython and Jupyter notebooks
- Interaction with external python code
- Some random hands on examples

### Data Preparation
- Some tasks on data handling and preparation
- Data loading and visualization on the CIFAR-10 dataset
- Splitting into training, validation and test sets
- Mean image subtraction

***

## Exercise 1
### Softmax
- Implementation of a fully-vectorized loss function for the Softmax classifier
- Implementation of the fully-vectorized expression for its analytic gradient
- Check of the implementation with numerical gradient
- Usage of a validation set to tune the learning rate and regularization strength
- Optimization of the loss function with SGD
- Visualization of the final learned weights

### Two layer net
- Implementation of a fully-connected two layer neural network to perform classification on the CIFAR-10 dataset
- Implementation of the forward and backward pass
- Training of the NN and hyperparameter training
- Visialization of the learned weights

### Features
- Improvement of the Two layer net by using extracted image features instead of raw image data
- Feature extraction: Histogram of Oriented Gradients (HOG) and color histogram using the hue channel ins HSV color space
- Training of the NN on the extracted features

***
