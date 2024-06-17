# Image Recognition using Pre-trained Neural Network

This project demonstrates how to use a pre-trained neural network model for image recognition tasks. By leveraging the power of models trained on large datasets, we can classify images with high accuracy without the need for extensive training.

## Table of Contents
- [Introduction](#introduction)
- [Script Description](#script-description)
- [Requirements](#requirements)
- [Setup](#setup)
- [Loading the Model](#loading-the-model)
- [Image Preprocessing](#image-preprocessing)
- [Loading and Displaying Images](#loading-and-displaying-images)
- [Model Inference](#model-inference)
- [Applications](#applications)
- [Acknowledgements](#acknowledgements)
  

## Introduction
This project utilizes a pre-trained neural network to perform image classification. Pre-trained models, such as VGG16, ResNet, and InceptionV3, are trained on extensive image datasets and can classify images into various categories with high accuracy

Vgg16 is a

## Script Description
The notebook performs the following tasks:

- Reads and Displays Images:
The script reads images named startup_example_scene1.png, startup_example_scene2.png, and startup_example_scene3.png using matplotlib.pyplot.imread.
Each image is displayed using matplotlib.pyplot.imshow.

- Prints Labels and Probabilities:
For each image, it prints predefined labels and probabilities from the P array.

## Requirements
Before running the script, we must ensure we have the following packages installed:

- `matplotlib`
- `numpy`
- `PIL` (Pillow)
- `scikit-image`
- `keras`


## Setup
First, we will import the necessary libraries and configure the environment. This includes setting up the TensorFlow and Keras environments, which are required for loading and using the pre-trained neural network model.

## Loading the Model
Next, we load our pre-trained VGG16 model, which is trained on the MNIST dataset. This model will be used to perform the image recognition tasks.

## Image Preprocessing
Then, we define a function to preprocess images before feeding them into the neural network. This preprocessing includes resizing the image to the required dimensions (224 x 224) , converting the image to an array, expanding its dimensions, and normalizing the pixel values to match the format expected by the model.

## Loading and Displaying Images
Load sample images from your local file system or a dataset. Display these images using Matplotlib to visually verify the inputs before they are processed by the model. This step ensures that the images are correctly loaded and formatted.

## Model Inference
Use the pre-trained model to predict the class of the loaded images. This involves feeding the preprocessed images into the model and obtaining the raw prediction results. The model outputs a set of probabilities for each class.
