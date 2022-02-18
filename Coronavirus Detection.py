#!/usr/bin/env python
# coding: utf-8

# ### Loading Libraries

# In[1]:


import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from PIL import Image
from tqdm import tqdm
from collections import Counter
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Model
from tensorflow.keras import applications
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import GlobalAveragePooling2D
from tensorflow.keras import optimizers
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import classification_report
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'


# ### Loading Datasets

# In[2]:


train_data = r"C:\Users\Chetan\Desktop\Covid19-dataset\train"
test_data = r"C:\Users\Chetan\Desktop\Covid19-dataset\test"


# In[3]:


covid_images = [os.path.join(train_data, 'Covid', path) for path in os.listdir(train_data + '/Covid')]
normal_images = [os.path.join(train_data, 'Normal', path) for path in os.listdir(train_data + '/Normal')]
viral_pneumonia_images = [os.path.join(train_data, 'Viral Pneumonia', path) for path in os.listdir(train_data + '/Viral Pneumonia')]


# In[4]:


image = Image.open(covid_images[0])


# In[5]:


plt.imshow(np.array(image))
plt.show()


# In[6]:


#now we normalize the data

train_datagen = ImageDataGenerator(rescale=1. / 255)
test_datagen = ImageDataGenerator(rescale=1. / 255)


# In[7]:


#first generating images without augmentation for a baseline model

img_width, img_height = 224, 224
batch_size = 8

train_generator = train_datagen.flow_from_directory(train_data, target_size=(img_height, img_width), batch_size=batch_size, class_mode='categorical')
test_generator = test_datagen.flow_from_directory(test_data, target_size=(img_height, img_width), batch_size=batch_size, class_mode='categorical')


# In[8]:


#checking the class counts

from collections import Counter
print("The individual class count in train set is ", Counter(train_generator.classes))
print("The individual class count in test set is ", Counter(test_generator.classes))


# In[9]:


#now we will build a base model

base_model = tf.keras.applications.resnet.ResNet50(weights='imagenet', include_top=False)
global_avg_pooling = tf.keras.layers.GlobalAveragePooling2D()(base_model.output)
output = tf.keras.layers.Dense(3, activation='softmax')(global_avg_pooling)
model = tf.keras.Model(inputs=base_model.input, outputs=output)
optimizer = tf.keras.optimizers.SGD(lr=1e-3, momentum=0.9, decay=0.01)
model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])


# In[10]:


history = model.fit(train_generator, epochs=10, validation_data=test_generator, verbose=1)


# In[11]:


#now lets plot epoch vs loss

plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(range(10), history.history['loss'], label='train loss')
plt.plot(range(10), history.history['val_loss'], label='val loss')
plt.title('Baseline model: Epoch vs loss')
plt.xlabel('epoch')
plt.ylabel('loss')
plt.legend()
plt.subplot(1, 2, 2)
plt.plot(range(10), history.history['accuracy'], label='train accuracy')
plt.plot(range(10), history.history['val_accuracy'], label='val accuracy')
plt.title('Baseline model: Epoch vs accuracy')
plt.xlabel('epoch')
plt.ylabel('accuracy')
plt.legend()
plt.show()


# __Observations:__
# We see that our baseline model gives us an overfit model. The val_accuracy is very low compared to the train accuracy.

# In[12]:


#alright now let us try to add dropout layer for regularization

base_model = tf.keras.applications.resnet.ResNet50(weights='imagenet', include_top=False)
global_avg_pooling = tf.keras.layers.GlobalAveragePooling2D()(base_model.output)
dropout = tf.keras.layers.Dropout(rate=0.5)(global_avg_pooling) #added dropout
output = tf.keras.layers.Dense(3, activation='softmax')(dropout)
model = tf.keras.Model(inputs=base_model.input, outputs=output)
optimizer = tf.keras.optimizers.Adam(learning_rate=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-07)
model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])


# In[13]:


history = model.fit(train_generator, epochs=10, validation_data=test_generator, verbose=1)


# In[14]:


#plotting epoch vs loss and accuracy again

plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(range(10), history.history['loss'], label='train loss')
plt.plot(range(10), history.history['val_loss'], label='val loss')
plt.title('Baseline model with dropout: Epoch vs loss')
plt.xlabel('epoch')
plt.ylabel('loss')
plt.legend()
plt.subplot(1, 2, 2)
plt.plot(range(10), history.history['accuracy'], label='train accuracy')
plt.plot(range(10), history.history['val_accuracy'], label='val accuracy')
plt.title('Baseline model with dropout: Epoch vs accuracy')
plt.xlabel('epoch')
plt.ylabel('accuracy')
plt.legend()
plt.show()


# __Obvservations:__
# We added dropout we see a slight improvement compared to the previous model, but still not upto the mark. To further fix overfitting we need to perform data augmentation so our train data size is increased.

# In[15]:


#performing data augmentation

train_datagen = ImageDataGenerator(rescale=1. / 255, 
                                   shear_range=0.2,
                                   zoom_range=0.2,
                                   width_shift_range=0.2,
                                   height_shift_range=0.2,
                                   horizontal_flip=True)
test_datagen = ImageDataGenerator(rescale=1. / 255)


# In[17]:


#generating images with augmentation for the model
img_width, img_height = 224, 224
batch_size = 8

train_generator = train_datagen.flow_from_directory(train_data, target_size=(img_height, img_width), batch_size=batch_size, class_mode='categorical')
test_generator = test_datagen.flow_from_directory(test_data, target_size=(img_height, img_width), batch_size=batch_size, class_mode='categorical')


# In[18]:


x, y = next(train_generator)


# In[19]:


x.shape


# In[20]:


for i, j in zip(range(1,9), range(0,8)):
    plt.subplot(2, 4, i)
    plt.imshow(x[j])


# In[21]:


base_model = tf.keras.applications.resnet.ResNet50(weights='imagenet', include_top=False)
global_avg_pooling = tf.keras.layers.GlobalAveragePooling2D()(base_model.output)
dropout = tf.keras.layers.Dropout(rate=0.5)(global_avg_pooling) #added dropout
output = tf.keras.layers.Dense(3, activation='softmax')(dropout)
model = tf.keras.Model(inputs=base_model.input, outputs=output)
optimizer = tf.keras.optimizers.Adam(learning_rate=1e-5, beta_1=0.9, beta_2=0.999, epsilon=1e-07)
model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])


# In[22]:


history = model.fit(train_generator, epochs=60, validation_data=test_generator, verbose=1)


# In[23]:


plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(range(60), history.history['loss'], label='train loss')
plt.plot(range(60), history.history['val_loss'], label='val loss')
plt.title('Model with Augmentation: Epoch vs loss')
plt.xlabel('epoch')
plt.ylabel('loss')
plt.legend()
plt.subplot(1, 2, 2)
plt.plot(range(60), history.history['accuracy'], label='train accuracy')
plt.plot(range(60), history.history['val_accuracy'], label='val accuracy')
plt.title('Model with Augmentation: Epoch vs accuracy')
plt.xlabel('epoch')
plt.ylabel('accuracy')
plt.legend()
plt.show()


# After running the model for 60 epochs, we get an excellent accuracy score of 100% on validation data. Further, we are going to check other metrics like confucion matrix and F1 score

# In[24]:


all_batches = []
count = 1
for batch in tqdm(test_generator):
    all_batches.append(batch)
    count = count + 1
    if count == 33:
        break


# In[25]:


len(all_batches)


# In[26]:


all_y_hats = []
all_y = []
for X, y in all_batches:
    y_hat = model.predict(X)
    y_hat = np.argmax(y_hat, 1)
    all_y_hats.extend(list(y_hat))
    all_y.extend(list(np.argmax(y, 1)))


# In[27]:


print(classification_report(all_y, all_y_hats))


# In[28]:


sns.heatmap(confusion_matrix(all_y, all_y_hats), annot=True, cmap='mako')
plt.title('Confusion Matrix on Test Set')
plt.show()


# ## Conclusion
# 
# Trained a convolutional neural network with ResNet-50 weights on Imagenet through transfer learning. The accuracy score achieved on validation set is 96.97%. Also, by looking at the confusion matrix, we see that principal diagnoal elements are very high and others are zero. Therefore, all of the data points in the test set are being classified correctly.
