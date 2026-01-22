# CIFAR-10 Image Classifier using CNN

## Project Overview
This project implements a **Convolutional Neural Network (CNN)** to classify images from the **CIFAR-10 dataset**, which contains 60,000 32x32 color images in 10 different classes, such as airplane, automobile, bird, cat, dog, etc. The model is trained using **TensorFlow/Keras** and can predict the class of new images.

## Features
- Trains a CNN with 3 convolutional layers and 2 max-pooling layers.
- Classifies images into 10 CIFAR-10 classes.
- Supports loading a trained model to predict new images.
- Visualizes both training samples and predictions using Matplotlib.
- Saves the trained model for future use.

## Dataset
- **CIFAR-10** dataset from `tensorflow.keras.datasets`.
- 50,000 training images and 10,000 testing images.
- Classes: airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck.

## Requirements
- Python 3.x
- Libraries:
  ```bash
  numpy
  matplotlib
  opencv-python
  tensorflow
