
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd

# Importing the Keras libraries and packages
from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense, Dropout,ZeroPadding2D
from keras.preprocessing.image import ImageDataGenerator,load_img, img_to_array
from keras.preprocessing import image


# In[2]:


# Initialising the CNN
classifier = Sequential()

# Step 1 - Convolution
classifier.add(Conv2D(32, (3, 3), input_shape = (124, 124, 3), activation = 'relu'))
classifier.add(ZeroPadding2D((1,1)))
classifier.add(Conv2D(32, (3, 3), activation = 'relu'))
classifier.add(ZeroPadding2D((1,1)))
classifier.add(MaxPooling2D(pool_size = (2, 2)))
classifier.add(ZeroPadding2D((1,1)))

# Adding a second convolutional layer
classifier.add(Conv2D(64, (3, 3), activation = 'relu'))
classifier.add(ZeroPadding2D((1,1)))
classifier.add(Conv2D(64, (3, 3), activation = 'relu'))
classifier.add(ZeroPadding2D((1,1)))
classifier.add(MaxPooling2D(pool_size = (2, 2)))
classifier.add(ZeroPadding2D((1,1)))

# Adding a third convolutional layer
classifier.add(Conv2D(64, (3, 3), activation = 'relu'))
classifier.add(ZeroPadding2D((1,1)))
classifier.add(Conv2D(64, (3, 3), activation = 'relu'))
classifier.add(ZeroPadding2D((1,1)))
classifier.add(MaxPooling2D(pool_size = (2, 2)))
classifier.add(ZeroPadding2D((1,1)))
classifier.add(Dropout(0.25))

# Adding a fourth convolutional layer
classifier.add(Conv2D(128, (3, 3), activation = 'relu'))
classifier.add(ZeroPadding2D((1,1)))
classifier.add(Conv2D(128, (3, 3), activation = 'relu'))
classifier.add(ZeroPadding2D((1,1)))
classifier.add(MaxPooling2D(pool_size = (2, 2)))
classifier.add(ZeroPadding2D((1,1)))
classifier.add(Dropout(0.25))

# Flatten the layer
classifier.add(Flatten())

# Step 4 - Full connection
classifier.add(Dense(units = 500, activation = 'relu'))
classifier.add(Dropout(0.5))
classifier.add(Dense(units = 11, activation = 'sigmoid'))

# Compiling the CNN
classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])


# In[3]:


classifier.summary()


# In[4]:


classifier.optimizer


# In[5]:


batch_size = 10
# this is the augmentation configuration we will use for training
train_datagen = ImageDataGenerator(rescale=1./255,
                                   shear_range=0.2,
                                   zoom_range=0.2,
                                   horizontal_flip=True)

# this is the augmentation configuration we will use for testing:
# only rescaling
valid_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory('F:/ML/clg/Dataset/train/',
                                                    # this is the target directory
                                                    target_size=(124, 124),  # all images will be resized to 150x150
                                                    batch_size=batch_size,
                                                    class_mode='categorical')  # more than two classes

validation_generator = valid_datagen.flow_from_directory('F:/ML/clg/Dataset/valid/',
                                                        target_size=(124,124),
                                                        batch_size=batch_size,
                                                        class_mode='categorical')


# In[6]:


# Fit the Classifier
classifier.fit_generator(train_generator,
                         steps_per_epoch = 10762,
                         epochs = 10,
                         validation_data = validation_generator,
                         validation_steps = 1108)


# In[7]:


classifier.get_weights()


# In[8]:


classifier.save('classifier.h5')


# In[1]:


from keras.preprocessing import image
test_image = image.load_img(r"C:\Users\Rashid Malik\Desktop\n02374451_16525.jpeg",
                            target_size = (124,124))
test_image = image.img_to_array(test_image)
test_image/=255
test_image = np.expand_dims(test_image, axis = 0)
result = classifier.predict(test_image)
print(train_generator.class_indices)
print(result)

