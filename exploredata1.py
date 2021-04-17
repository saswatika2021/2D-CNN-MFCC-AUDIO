#we have deeplearning, better hardware and more open source of data
# we selected emotion our target because we are dealing conversation between agent and custmor and there's variety of sources.
# Surrey Audio-Visual Expressed Emotion (SAVEE)
#that they are all male speakers only
#get the data location for SAVEE
# Import libraries 
import librosa
import librosa.display
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from matplotlib.pyplot import specgram
import pandas as pd
import glob 
from sklearn.metrics import confusion_matrix
import IPython.display as ipd  # To play sound in the notebook
import os
import sys
import warnings
from os.path import isfile, join

# ignore warnings 
if not sys.warnoptions:
    warnings.simplefilter("ignore")
warnings.filterwarnings("ignore", category=DeprecationWarning) 

SAVEE = "C:\\Users\\saswa\\AI_ML\\Audio-Sentiment-Analysis\\data\\AudioData\\AudioData\\DC\\"
new_path = "C:\\Users\\saswa\\AI_ML\\Audio-Sentiment-Analysis\\data\\original_data\\DC"
path = "C:\\Users\\saswa\\AI_ML\\Audio-Sentiment-Analysis\\data\\original_data\\AudioData\\"


import shutil

root_src_dir = 'C:\\Users\\saswa\\AI_ML\\Audio-Sentiment-Analysis\\data\\original_data\\'
root_dst_dir = 'C:\\Users\\saswa\\AI_ML\\Audio-Sentiment-Analysis\\data\\original_data\\AudioData\\'



""" 
Renames the filenames within the same directory to be Unix friendly
(1) Changes spaces to hyphens
(2) Makes lowercase (not a Unix requirement, just looks better ;)
Usage:
python rename.py
"""

path =  os.getcwd()
filenames = os.listdir(path)

for filename in filenames:
    os.rename(filename, filename.replace("_", ""))
    
    
'''        
# parse the filename to get the emotions
emotion=[]
path = []
for i in dir_list:
    if i[-8:-6]=='_a':
        emotion.append('male_angry')
    elif i[-8:-6]=='_d':
        emotion.append('male_disgust')
    elif i[-8:-6]=='_f':
        emotion.append('male_fear')
    elif i[-8:-6]=='_h':
        emotion.append('male_happy')
    elif i[-8:-6]=='_n':
        emotion.append('male_neutral')
    elif i[-8:-6]=='sa':
        emotion.append('male_sad')
    elif i[-8:-6]=='su':
        emotion.append('male_surprise')
    else:
        emotion.append('male_error') 
        path.append(SAVEE + i)
   
         
        
#Separating audio clips on the basis of 8 emotions and storing them together accordingly
#Emotion (01 = neutral, 02 = calm, 03 = happy, 04 = sad, 05 = angry, 06 = fearful, 07 = disgust, 08 = surprised)


#Separating audio clips on the basis of 8 emotions and storing them together accordingly
#Emotion (01 = neutral, 02 = calm, 03 = happy, 04 = sad, 05 = angry, 06 = fearful, 07 = disgust, 08 = surprised)

        
    

# parse the file to get emmotion
    

    
#Ryerson Audio-Visual Database of Emotional Speech and Song (RAVDESS)
#Toronto emotional speech set (TESS)
#Crowd-sourced Emotional Mutimodal Actors Dataset (CREMA-D)

'''
