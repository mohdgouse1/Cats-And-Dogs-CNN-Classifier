# Convolutional Neural Network

# Installing Theano
# pip install --upgrade --no-deps git+git://github.com/Theano/Theano.git

# Installing Tensorflow
# Install Tensorflow from the website: https://www.tensorflow.org/versions/r0.12/get_started/os_setup.html

# Installing Kerassa
# pip install --upgrade keras

# Part 1 - Building the CNN

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning) 

# Importing the Keras libraries and packages
from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense
import os

from tensorflow.python.client import device_lib
print(device_lib.list_local_devices())


from keras import backend as K
K.tensorflow_backend._get_available_gpus()

os.environ['KMP_DUPLICATE_LIB_OK']='True'

# Initialising the CNN
classifier = Sequential()

# Step 1 - Convolution  
classifier.add(Conv2D(filters = 32, kernel_size = 3, input_shape = (64, 64, 3), activation = 'relu'))

# Step 2 - Pooling
classifier.add(MaxPooling2D(pool_size = (2, 2)))

# Adding a second convolutional layer
classifier.add(Conv2D(filters = 32, kernel_size = 3, activation = 'relu'))
classifier.add(MaxPooling2D(pool_size = (2, 2)))

# Step 3 - Flattening
classifier.add(Flatten())

# Step 4 - Full connection
classifier.add(Dense(units = 128, activation = 'relu'))
classifier.add(Dense(units = 1, activation = 'sigmoid'))

# Compiling the CNN
classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

# Part 2 - Fitting the CNN to the images

from keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(rescale = 1./255,
                                   shear_range = 0.2,
                                   zoom_range = 0.2,
                                   horizontal_flip = True)

test_datagen = ImageDataGenerator(rescale = 1./255)

training_set = train_datagen.flow_from_directory('training_set',
                                                 target_size = (64, 64),
                                                 batch_size = 32,
                                                 class_mode = 'binary')

test_set = test_datagen.flow_from_directory('test_set',
                                            target_size = (64, 64),
                                            batch_size = 32,
                                            class_mode = 'binary')

classifier.fit_generator(training_set,
                         steps_per_epoch = 8000,
                         epochs = 5,
                         validation_data = test_set,
                         validation_steps = 2000)


#Predictions

import numpy as np
from keras.preprocessing  import image

test_image = image.load_img('Prediction/cat_or_dog_2.jpg', target_size= (64,64))
test_image = image.img_to_array(test_image)
test_image = np.expand_dims(a = test_image, axis = 0)
result = classifier.predict(test_image)

training_set.class_indices

if result[0][0] == 1:
    print("DOG")
elif result[0][0] == 0:
        print("CAT")
else:
    print("Unclassified")


