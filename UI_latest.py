#!/usr/bin/env python
# coding: utf-8

# In[1]:


get_ipython().system('pip3 install tensorflow')
get_ipython().system('pip3 install opencv-python keras')
import pandas as pd

import re
import cv2
import numpy as np
from keras.models import load_model


# In[2]:



import keras
from matplotlib import pyplot as plt
import numpy as np
import gzip
get_ipython().run_line_magic('matplotlib', 'inline')
from keras.models import Model
from keras.optimizers import RMSprop
from keras.layers import Input,Dense,Flatten,Dropout,merge,Reshape,Conv2D,MaxPooling2D,UpSampling2D,Conv2DTranspose
from keras.layers.normalization import BatchNormalization
from keras.models import Model,Sequential
from keras.callbacks import ModelCheckpoint
from keras.optimizers import Adadelta, RMSprop,SGD,Adam
from keras import regularizers
from keras import backend as K
from keras.utils import to_categorical
def generate_vgg16(num_classes, in_shape = (224,224,3)):
    model = Sequential()
    model.add(Conv2D(32, (3, 3), activation='relu', padding='same',name='block1_conv1', input_shape=in_shape))
    model.add(Conv2D(32, (3, 3), activation='relu', padding='same', name='block1_conv2'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2), name='block1_pool'))

    # Block 2
    model.add(Conv2D(64, (3, 3), activation='relu', padding='same', name='block2_conv1'))
    model.add(Conv2D(64, (3, 3), activation='relu', padding='same', name='block2_conv2'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2), name='block2_pool'))

    # Block 3
    model.add(Conv2D(128, (3, 3), activation='relu', padding='same', name='block3_conv1'))
    model.add(Conv2D(128, (3, 3), activation='relu', padding='same', name='block3_conv2'))
    model.add(Conv2D(128, (3, 3), activation='relu', padding='same', name='block3_conv3'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2), name='block3_pool'))

    # Block 4
    model.add(Conv2D(256, (3, 3), activation='relu', padding='same', name='block4_conv1'))
    model.add(Conv2D(256, (3, 3), activation='relu', padding='same', name='block4_conv2'))
    model.add(Conv2D(256, (3, 3), activation='relu', padding='same', name='block4_conv3'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2), name='block4_pool'))

    # Classification block
    model.add(Flatten(name='flatten'))
    model.add(Dense(512, activation='relu', name='fc1',kernel_regularizer=regularizers.l1_l2(l1=1e-5, l2=1e-4),
    bias_regularizer=regularizers.l2(0.03),
    activity_regularizer=regularizers.l2(0.03)))
    model.add(Dropout(0.8))
    model.add(Dense(512, activation='relu', name='fc2',kernel_regularizer=regularizers.l1_l2(l1=1e-5, l2=1e-4),
    bias_regularizer=regularizers.l2(0.05),
    activity_regularizer=regularizers.l2(0.05)))
    model.add(Dropout(0.8))
    model.add(Dense(num_classes, activation='softmax', name='predictions'))

    return model

race_model=generate_vgg16(7)

opt = keras.optimizers.RMSprop(lr = 0.00008, decay = 1e-6)
race_model.compile(loss=keras.losses.categorical_crossentropy, optimizer=opt,metrics=['accuracy'])
race_model.load_weights('53.h5')


# In[3]:


import keras
from matplotlib import pyplot as plt
import numpy as np
import gzip
get_ipython().run_line_magic('matplotlib', 'inline')
from keras.models import Model
from keras.optimizers import RMSprop
from keras.layers import Input,Dense,Flatten,Dropout,merge,Reshape,Conv2D,MaxPooling2D,UpSampling2D,Conv2DTranspose
from keras.layers.normalization import BatchNormalization
from keras.models import Model,Sequential
from keras.callbacks import ModelCheckpoint
from keras.optimizers import Adadelta, RMSprop,SGD,Adam
from keras import regularizers
from keras import backend as K
from keras.utils import to_categorical
def generate_vgg16(num_classes, in_shape = (224,224,3)):
    model = Sequential()
    model.add(Conv2D(32, (3, 3), activation='relu', padding='same',name='block1_conv1', input_shape=in_shape))
    model.add(Conv2D(32, (3, 3), activation='relu', padding='same', name='block1_conv2'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2), name='block1_pool'))

    # Block 2
    model.add(Conv2D(64, (3, 3), activation='relu', padding='same', name='block2_conv1'))
    model.add(Conv2D(64, (3, 3), activation='relu', padding='same', name='block2_conv2'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2), name='block2_pool'))

    # Block 3
    model.add(Conv2D(128, (3, 3), activation='relu', padding='same', name='block3_conv1'))
    model.add(Conv2D(128, (3, 3), activation='relu', padding='same', name='block3_conv2'))
    model.add(Conv2D(128, (3, 3), activation='relu', padding='same', name='block3_conv3'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2), name='block3_pool'))

    # Block 4
    model.add(Conv2D(256, (3, 3), activation='relu', padding='same', name='block4_conv1'))
    model.add(Conv2D(256, (3, 3), activation='relu', padding='same', name='block4_conv2'))
    model.add(Conv2D(256, (3, 3), activation='relu', padding='same', name='block4_conv3'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2), name='block4_pool'))

    # Classification block
    model.add(Flatten(name='flatten'))
    model.add(Dense(512, activation='relu', name='fc1',kernel_regularizer=regularizers.l1_l2(l1=1e-5, l2=1e-4),
    bias_regularizer=regularizers.l2(0.01),
    activity_regularizer=regularizers.l2(0.01)))
    model.add(Dropout(0.8))
    model.add(Dense(512, activation='relu', name='fc2',kernel_regularizer=regularizers.l1_l2(l1=1e-5, l2=1e-4),
    bias_regularizer=regularizers.l2(0.01),
    activity_regularizer=regularizers.l2(0.01)))
    model.add(Dropout(0.8))
    model.add(Dense(num_classes, activation='softmax', name='predictions'))

    return model

race_modelm=generate_vgg16(7)


# In[4]:


optm = keras.optimizers.RMSprop(lr = 0.00008, decay = 1e-6)
race_modelm.compile(loss=keras.losses.categorical_crossentropy, optimizer=optm,metrics=['accuracy'])
race_modelm.load_weights('58_Mask_55_2.h5')


# In[5]:


sample_face = cv2.imread(r"C:\Users\sudhamshu\Desktop\val\12.jpg")
sample_face = cv2.resize(sample_face, (64,64) )


# In[22]:


model = load_model('gender_model_64.h5')


# In[23]:


def open_img(path):
    print(type(path))
    if(type(path)!=type('yu')):
        path=cv2.cvtColor(path, cv2.COLOR_BGR2RGB)

        img=Image.fromarray(path)
        #img.show()

        # resize the image and apply a high-quality down sampling filter
        img = img.resize((224, 224), Image.ANTIALIAS)

        # PhotoImage class is used to add image to widgets, icons etc'
    else:
        img=Image.open(path)
        img = img.resize((224, 224), Image.ANTIALIAS)
        
    img = ImageTk.PhotoImage(img)
        
   
    # create a label
    panel = Label(root, image = img)
      
    # set the image as img 
    panel.image = img
    panel.grid(row = 4)


# In[24]:


def read_img_and_preprocess_and_predict(path):
  open_img(path)
  testing_im=[]
  
  face = cv2.imread(path)
  
  testing_im.append(sample_face)
  face = cv2.imread(path)
  face = cv2.resize(face, (64,64) )
  testing_im.append(face)

  testing_imggg=np.squeeze(testing_im)
  tpp = testing_imggg.astype('float32') 
  tpp /= 255

  pre_im = model.predict(tpp)
  
  if pre_im[1][0]>=pre_im[1][1]:
    return ("Male", pre_im[1][0])
  else:
    return ("Female", pre_im[1][1])


# In[25]:


from retinaface import RetinaFace
import matplotlib.pyplot as plt


# In[26]:


import numpy as np
import cv2
def read_img_and_preprocess_and_predict_race(path):
  #open_img(path)
 
  
  
  
  testing_im=[]
  
  labels =['Black', 'East Asian', 'Indian', 'Latino_Hispanic','Middle Eastern', 'Southeast Asian', 'White']
    
  #testing_im.append(sample_face_race)
  face = cv2.imread(path)
  face = cv2.imread(path)
  faces = RetinaFace.extract_faces(path, align = True)
  for face in faces:
        plt.imshow(face[...,::-1])
        plt.show()
  try:
        face=faces[0]

        print('cropped')
  except:
        face=face
  
  img = face
 
    
  width =224
  height =224
  dim = (width, height)
  

  resized = cv2.resize(img, dim, interpolation = cv2.INTER_AREA)  
  face1=resized
  
  face = cv2.resize(face, (224,224) )
  
  open_img(face1)
  testing_im.append(face)
  
  testing_imggg=np.squeeze(testing_im)
  tpp = testing_imggg.astype('float32') 
  tpp /= 255
  f1=list()
  f1.append(face1)
  f1=np.array(f1)
  pre_im_race = race_model.predict(f1)
  p=pre_im_race[0]
  p=list(p)
  max_output= max(p)
  return (labels[p.index(max_output)])

#   if pre_im_race[1][0]>=pre_im[1][1]:
#     return ("Male", pre_im[1][0])
#   else:
#     return ("Female", pre_im[1][1])

#read_img_and_preprocess_and_predict_race(r"C:\Users\sudhamshu\Pictures\IMG_20200927_195552.jpg")


# In[27]:


def read_img_and_preprocess_and_predict_race_mask(path):
  #open_img(path)
  testing_im=[]
  
  labels =['Black', 'East Asian', 'Indian', 'Latino_Hispanic',
       'Middle Eastern', 'Southeast Asian', 'White']
    
  face = cv2.imread(path)
  faces = RetinaFace.extract_faces(path, align = True)
  for face in faces:
        plt.imshow(face[...,::-1])
        plt.show()
  try:
        face=faces[0]

        print('cropped')
  except:
        face=face
  
  img = face
    
  width =224
  height =224
  dim = (width, height)
  

  resized = cv2.resize(img, dim, interpolation = cv2.INTER_AREA)  
  face1=resized
  #cv2.imshow('face',face1)
  face = cv2.resize(face, (224,224) )
  open_img(face1)
  testing_im.append(face)
  
  testing_imggg=np.squeeze(testing_im)
  tpp = testing_imggg.astype('float32') 
  tpp /= 255
  f1=list()
  f1.append(face1)
  f1=np.array(f1)
  pre_im_race = race_modelm.predict(f1)
  p=pre_im_race[0]
  p=list(p)
  max_output= max(p)
  return (labels[p.index(max_output)])


# In[28]:


from tkinter import *
from PIL import ImageTk, Image
import os
from tkinter import filedialog
root = Tk()

#####  Title
root.title("Predict gender & race")

root.geometry("780x640")
  
root.resizable(width = True, height = True)


#label2 = Label(root, text = "Enter the absolute path of the image",font = "Arial 20 bold", fg = 'black')
#label2.grid(row = 2, column = 0)

##### Text box to enter the question
#ques = StringVar()
#quesEntered = Entry(root, width = 100, textvariable = ques,font="Arial 18 bold")
#quesEntered.grid(column = 0, row = 3, padx = 3, pady = 3)


# function that takes the question from user
# predicts the tags to the questions
def openfilename():
  
    # open file dialog box to select image
    # The dialogue box has a title "Open"
    filename = filedialog.askopenfilename(title ='"pen')
    return filename
x=[]
def open_img1():
    # Select the Imagename  from a folder 
    x.clear()
    x.append(openfilename())
   
    open_img(x[0])

  

ans = StringVar()
global img 
def pred_gender():
    #t = ques.get()
    res= read_img_and_preprocess_and_predict(x[0])
    ans.set(res)
    label3 = Entry(root,textvariable = ans, font = "Arial 20 bold")
    label3.grid(column =0, row = 5, padx = 3, pady = 3)

def pred_race():
    #t= ques.get()
    res= read_img_and_preprocess_and_predict_race(x[0])
    ans.set(res)
    label3 = Entry(root,textvariable = ans, font = "Arial 20 bold")
    label3.grid(column =0, row = 5, padx = 3, pady = 3)

def pred_race_mask():
    #t= ques.get()
    res= read_img_and_preprocess_and_predict_race_mask(x[0])
    #print(x[0])
    ans.set(res)
    label3 = Entry(root,textvariable = ans, font = "Arial 20 bold")
    label3.grid(column =0, row = 5, padx = 3, pady = 3)

##### Button that predicts the tags by calling the function predict
label2 = Label(root, text = "Select the image from the menu by pressing select image",font = "Arial 20 bold", fg = 'black')
label2.grid(row = 1, column = 0, padx = 3, pady = 3)

btn = Button(root, text ='select image', font = "Arial 18 bold",command = open_img1).grid(
                                        row = 3, columnspan = 4,padx=20, pady=40)
button = Button(root,text = "Predict Gender", font = "Arial 18 bold", command = pred_gender)
button2 = Button(root,text = "Predict Race", font = "Arial 18 bold", command = pred_race)
button3 = Button(root,text = "Predict Race with mask", font = "Arial 18 bold", command = pred_race_mask)
button.grid(column= 0, row = 6, padx = 3, pady = 3)
button2.grid(column= 0, row = 7, padx = 3, pady = 3)
button3.grid(column= 0, row = 8, padx = 3, pady = 3)



root.mainloop()


# In[ ]:





# In[ ]:





# In[ ]:




