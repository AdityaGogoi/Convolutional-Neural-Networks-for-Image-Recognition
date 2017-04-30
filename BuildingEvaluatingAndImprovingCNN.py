# -*- coding: utf-8 -*-
"""
Created on Tue Apr 18 11:57:58 2017

@author: Aditya Gogoi
"""

        # Convolutional Neural Network
        # Part 1 - Building the CNN
# We wont be requiring any Data Preprocessing as we have already split the dataset into an 80:20  split of Train and Test dataset.
# But we will be requiring Feature scaling, which we will do just before we train our CNN on the dataset

# Importing the Keras libraries and packages
# 1. Sequential - To initialise our Neural Network (as a sequence of layers)
# 2. Convolution2D - To perform the convolution step of CNN (the 2D is because images are 2D).
# 3. MaxPooling2D - To add the pooling layers for the CNN.
# 4. Flatten - To convert the images as input for NN.
# 5. Dense - To create our fully connected NN from the Layers.
from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense

# Initialising the CNN, just like the ANN
classifier = Sequential()

    # Step 1 - Convolution
# This layer will be added like just any other layer of nodes in the NN.
# We sue the Conv2D class with these features:-
# 1. Number of filters / feature detectors.
# 2. Number of (rows,columns) in each filter.
# 3. border_area = we will ignore this as it will be assumed as default value.
# 4. input_shape = Consists of the parameters of the image that we will input because images might vary and we have to provide them even-ground.
#       Parameters for input_shape are:-
#       1. The number of dimensions in the image. For color, there are 3 dimensions (red, green, blue) and for b&w image we take 2 dimensions.
#       2. Number of pixels we need to retain (x,y). Default is (256, 256), but we can change it based on pur machines speed. We will keep it (64, 64).
#       For tensorflow, the sequence of parameters is opposite i.e. Pixels first, then dimensions.
# 5. The activation function. We will be using the Rectifier function in order to remove the linearity in the Map.
classifier.add(Conv2D(32, (3, 3), input_shape = (64, 64, 3), activation = 'relu'))

    # Step 2 - Pooling
# This will reduce the size of our image without affecting the performance.
# The max pooling step gets the maximum within the window / filter on the image.
# We use the class MaxPooling2D with the following features:-
# 1. pool_size = The size of the filter window to be used. For better accuracy, we will use pool size of 2*2.
classifier.add(MaxPooling2D(pool_size = (2, 2)))

# Adding another Convolutional Layer for higher accuracy.
# Excluding input_shape because we already ahve the pooled layer and we know what the input is.
classifier.add(Conv2D(32, (3, 3), activation = 'relu'))
classifier.add(MaxPooling2D(pool_size = (2, 2)))

    # Step 3 - Flattening
# To convert the Pool maps into a single dimensional vector to be used as input for the NN.
# We use the Flatten class to implement it.
classifier.add(Flatten())

    # Step 4 - Full Connection
# This is the part where the Fully Connected NN will receive the Flattened input and make predictions.
# We will keep the output node as 1 because the output is binary (Cat or Dog).
# The hidden layer is first created using the Dense class with following parameters:-
# 1. output_dim = Contains number of nodes in the hidden layer. Has to be large because of the flatten input. The number is based on trial and can be kept as a power of 2.
# 2. activation = Is the activation function used for the hidden layer. We are using Rectifier function.
# For the output layer, we will keep the number of nodes as 1 and the activation function as Sigmoid, because of the binary nature of the output. For multiclass classification, we shud use Softmax function.
classifier.add(Dense(output_dim = 128, activation = 'relu')) 
classifier.add(Dense(output_dim = 1, activation = 'sigmoid'))

# Compiling the CNN
# Same as the step for ANN.
classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

        # Part 2 - Fitting the CNN to the images
# The dataset we have is not large enough to train the CNN. This will lead to overfitting, and the CNN will not be able to generalize the correlations that it has observed between the independant and dependant atrributes.
# For this reason we perform Image Augmentation during the image preprocessing of our dataset.
# Image augmentation will take our existing dataset and add a few transformations to it to make new images, thus increasing it's size.
# This can be found in the Keras documentation under Image Preprocessing section.
# 1. ImageDataGenerator - This will change th features of the images by shearing, zooming, etc.
# 2. flow_from_directory - This will collect the images from the respective directory.
# 3. fit_generator - This will fit our model with the dataset and provide accuracy. 
from keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(rescale = 1./255,
                                   shear_range = 0.2,
                                   zoom_range = 0.2,
                                   horizontal_flip = True)

test_datagen = ImageDataGenerator(rescale = 1./255)

training_set = train_datagen.flow_from_directory('dataset/training_set',
                                                 target_size = (64, 64),
                                                 batch_size = 32,
                                                 class_mode = 'binary')

test_set = test_datagen.flow_from_directory('dataset/test_set',
                                            target_size = (64, 64),
                                            batch_size = 32,
                                            class_mode = 'binary')

classifier.fit_generator(training_set,
                         steps_per_epoch = 8000,
                         epochs = 25,
                         validation_data = test_set,
                         validation_steps = 2000)

        # Part 3 - Making New predictions

import numpy as np
from keras.preprocessing import image

# Getting the image into specific size being used by the CNN
test_image = image.load_image('dataset/single_prediction/cat_or_dog.jpg',
                              target_size=(64,64))
# Converting the image into an array of 3 dimensions (64*64*3). The 3 is for the 3 colors i.e. red,blue,green.
test_image = image.img_to_array(test_image)
# The NN is used for working on batches of inputs, the inputs can be just 1 or several.
# This additional "batch" occurs as the additional dimension in the CNN input. The CNN will not work if this additional dimension is not present within the input.
# Since we now have just 1 input image, we will add this additional dimension using "expand_dims" function. 
# The first argument for this function will be the input, and the next will be along which axis should the extra dimension be added. In this case, its along axis 0 i.e. 1 more column.
test_image = np.expand_dims(test_image, axis=0)
# Storing the prediction in 'result' variable.
result=classifier.predict(test_image)
# To check what 0 or 1 in the result represent, we will see how the CNN classifies them.
training_set.class_indices
# We can return a string value once we know the boolean value of cat and dog.
'''
if result[0][0]==<the boolean value for dog>:
    prediction = 'dog'
else:
    prediction = 'cat'
'''
# Repeat the same with test_image as the next image in the folder.



