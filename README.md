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

## Exercise 2
### Fully Connected Nets
- Implementation of a modular fully connected neural network
- Affine layer: forward and backward
- ReLU layer: forward and backward
- Sandwich layers: affine-relu
- Loss layers: Softmax
- Implementation of a solver class to run the training process decoupled from the network model
- Implementation of different update rules: SGD, SGD+Momentum, Adam
- Hyperparameter tuning and model training

### Batch Normalization
- Implementation of a batch normalization layer
- Training of a network with batch normalization
- Comparison of different weight initializations and the interaction with batchnorm

### Dropout
- Implementaion of a dropout layer

***

## Exercise 3
### Semantic Segmentation
- Working with MSRC-v2 Segmentation Dataset
- Initialize training and validation data loaders.
- Design and initialize a convolutional neural network architecture and is based on an already pretrained network.
- Initialize a solver with a loss function that considers the unlabeled pixels.
- Adjusting the solver to account for the unlabeled pixels.
- Hyperparameter tuning and model training

### Facial Keypoint Detection
- Loading and visualizing Data
- Preprocessing transforms
- Implementing NaimishNet (https://arxiv.org/pdf/1710.00977.pdf)
- Data iterating and batching
- Hyperparameter tuning and model training

### RNN and LSTM
- Exploring vanishing gradients
- Gradient comparison
- MNIST image classification with RNNs
