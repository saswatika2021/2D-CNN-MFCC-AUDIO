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
# Command for file prefix FOR /r "." %a in (*.*) DO REN "%~a" "prefix%~nxa"
SAVEE = "C:\\Users\\saswa\\AI_ML\\Audio-Sentiment-Analysis\\data\\original_data\\AudioData\\"
RAVDSS = "C:\\Users\\saswa\\AI_ML\\Audio-Sentiment-Analysis\\data\\audio_data\\"
TESS = "C:\\Users\\saswa\\AI_ML\\Audio-Sentiment-Analysis\\data\\dataverse_files\\TESS\\"
CREMA = "C:\\Users\\saswa\\AI_ML\\Audio-Sentiment-Analysis\\data\\CREMA-D\\AudioWAV\\"

dir_list = os.listdir(SAVEE)
#print(dir_list[0:5])
#result : ['DC_a01.wav', 'DC_a02.wav', 'DC_a03.wav', 'DC_a04.wav', 'DC_a05.wav']
       
# parse the filename to get the emotions
emotion=[]
path = []
for i in dir_list:
    if i[-8:-6] == "_a":
        emotion.append("male_angry")
    elif i[-8:-6] == "_d":
        emotion.append("male_disgust")
    elif i[-8:-6] == "_f":
        emotion.append("male_fear")
    elif i[-8:-6] == "_h":
        emotion.append("male_happy")
    elif i[-8:-6] == "_n":
        emotion.append("male_neutral")
    elif i[-8:-6] == "sa":
        emotion.append("male_sad")
    elif i[-8:-6] == "su":
        emotion.append("male_surprise")
    else:
        emotion.append("male_error")
    path.append(SAVEE + i)
# check out the label count distribution
SAVEE_df = pd.DataFrame(emotion, columns = ['labels'])

SAVEE_df['source'] = 'SAVEE'
SAVEE_df = pd.concat([SAVEE_df, pd.DataFrame(path,columns = ['path'])], axis = 1)
#print(SAVEE_df.labels.value_counts())
#print(SAVEE_df)
#http://help.nchsoftware.com/help/en/wavepad/win/concepts.html
fname = SAVEE + 'DC_f11.wav'
data,sampling_rate = librosa.load(fname)
plt.figure(figsize = (15,5))
librosa.display.waveplot(data, sr = sampling_rate)
ipd.Audio(fname)
# RAVDSS DATASET
# speakers, recording and it has 24 actors of different genders
#Modality (01 = full-AV, 02 = video-only, 03 = audio-only).
#Vocal channel (01 = speech, 02 = song).
#Emotion (01 = neutral, 02 = calm, 03 = happy, 04 = sad, 05 = angry, 06 = fearful, 07 = disgust, 08 = surprised).
#Emotional intensity (01 = normal, 02 = strong). NOTE: There is no strong intensity for the 'neutral' emotion.
#Statement (01 = "Kids are talking by the door", 02 = "Dogs are sitting by the door").
#Repetition (01 = 1st repetition, 02 = 2nd repetition).
#Actor (01 to 24. Odd numbered actors are male, even numbered actors are female).
#Video-only (02)
#Speech (01)
#Fearful (06)
#Normal intensity (01)
#Statement "dogs" (02)
#1st Repetition (01)
#12th Actor (12) - Female (as the actor ID number is even)

dir_list_RAV = os.listdir(RAVDSS)
dir_list_RAV.sort()
#f = dir_list_RAV[0:5]
# result ['Actor_01', 'Actor_02', 'Actor_03', 'Actor_04', 'Actor_05']

emotion = []
gender = []
path = []
for i in dir_list_RAV:
    fname = os.listdir(RAVDSS + i)
    for f in fname:
        part = f.split('.')[0].split('-')
        emotion.append(int(part[2]))
        temp = int(part[6])
        if temp%2 == 0:
            temp = "female"
        else:
            temp = "male"
        gender.append(temp)
        path.append(RAVDSS + i + '/' + f)
RAV_df = pd.DataFrame(emotion)

RAV_df = RAV_df.replace({1:'neutral', 2:'neutral', 3:'happy', 4:'sad', 5:'angry', 6:'fear', 7:'disgust', 8:'surprise'})
RAV_df = pd.concat([pd.DataFrame(gender),RAV_df],axis=1)
RAV_df.columns = ['gender','emotion']
RAV_df['labels'] =RAV_df.gender + '_' + RAV_df.emotion
RAV_df['source'] = 'RAVDESS'  
RAV_df = pd.concat([RAV_df,pd.DataFrame(path, columns = ['path'])],axis=1)
RAV_df = RAV_df.drop(['gender', 'emotion'], axis=1)
#print(RAV_df.labels.value_counts())
#print(RAV_df)
        
# result '03-01-01-01-01-02-23.wav', '03-01-01-01-02-01-23.wav', 
#f = "03-01-01-01-01-02-23.wav"
#part = f.split(".")[0].split("_")
#print(part)
#result 03-01-01-01-01-02-23
 
# parse the file to get emmotion
    

#Ryerson Audio-Visual Database of Emotional Speech and Song (RAVDESS)
#3. TESS dataset
#Toronto emotional speech set (TESS)
#a young female and an older female. 

dir_list = os.listdir(TESS)
dir_list.sort()

path = []
emotion = []
for i in dir_list:
    fname = os.listdir(TESS + i)
    for f in fname:
        if i == "OAF_angry" or i == "YAF_angry":
            emotion.append("female_angry")
        elif i == "OAF_disgust" or i == "YAF_disgust":
            emotion.append("female_disgust")
        elif i == "OAF_sad" or i == "YAF_sad":
            emotion.append("female_sad")
        elif i == "OAF_happy" or i == "YAF_happy":
            emotion.append("female_happy")
        elif i == "OAF_fear" or i == "YAF_fear":
            emotion.append("female_fear")
        elif i == "OAF_neutral" or i == "YAF_neutral":
            emotion.append("female_neutral")
        elif i == "OAF_Pleasant_Surprise" or i == "YAF_Pleasant_Surprise":
            emotion.append("female_ps")
        else:
            emotion.append("Unknown")
        path.append(TESS + i +"/" + f) 
TESS_df = pd.DataFrame(emotion,columns = ['labels'])
TESS_df['source'] = 'TESS'
TESS_df = pd.concat([TESS_df,pd.DataFrame(path, columns = ['path'])], axis = 1)
#print(TESS_df)
#print(TESS_df.labels.value_counts())
#Crowd-sourced Emotional Mutimodal Actors Dataset (CREMA-D)
dir_list = os.listdir(CREMA)
dir_list.sort()
#print(dir_list[0:10])
gender = []
emotion = []
path = []
female = [1002,1003,1004,1006,1007,1008,1009,1010,1012,1013,1018,1020,1021,1024,1025,1028,1029,1030,1037,1043,1046,1047,1049,
          1052,1053,1054,1055,1056,1058,1060,1061,1063,1072,1073,1074,1075,1076,1078,1079,1082,1084,1089,1091]
for i in dir_list: 
    part = i.split('_')
    if int(part[0]) in female:
        temp = 'female'
    else:
        temp = 'male'
    gender.append(temp)
    if part[2] == 'SAD' and temp == 'male':
        emotion.append('male_sad')
    elif part[2] == 'ANG' and temp == 'male':
        emotion.append('male_angry')
    elif part[2] == 'DIS' and temp == 'male':
        emotion.append('male_disgust')
    elif part[2] == 'FEA' and temp == 'male':
        emotion.append('male_fear')
    elif part[2] == 'HAP' and temp == 'male':
        emotion.append('male_happy')
    elif part[2] == 'NEU' and temp == 'male':
        emotion.append('male_neutral')
    elif part[2] == 'SAD' and temp == 'female':
        emotion.append('female_sad')
    elif part[2] == 'ANG' and temp == 'female':
        emotion.append('female_angry')
    elif part[2] == 'DIS' and temp == 'female':
        emotion.append('female_disgust')
    elif part[2] == 'FEA' and temp == 'female':
        emotion.append('female_fear')
    elif part[2] == 'HAP' and temp == 'female':
        emotion.append('female_happy')
    elif part[2] == 'NEU' and temp == 'female':
        emotion.append('female_neutral')
    else:
        emotion.append('Unknown')
    path.append(CREMA + i)
    
CREMA_df = pd.DataFrame(emotion, columns = ['labels'])

CREMA_df['source'] = 'CREMA'
CREMA_df = pd.concat([CREMA_df,pd.DataFrame(path, columns = ['path'])],axis=1)
#print(CREMA_df.labels.value_counts())
df = pd.concat([CREMA_df,SAVEE_df,TESS_df,RAV_df],ignore_index=True)
#print(df)
df = pd.DataFrame(df)
#print(df)
df.to_csv( r"C:\\Users\\saswa\\AI_ML\\Audio-Sentiment-Analysis\\data\\DATA_PATH.csv") 
