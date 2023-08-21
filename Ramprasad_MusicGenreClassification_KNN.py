
#The dataset used here is the GTZAN, which can be downloaded from https://www.kaggle.com/datasets/andradaolteanu/gtzan-dataset-music-genre-classification and the features are in the file, genres_original, in GTZAN.

from python_speech_features import mfcc
import scipy.io.wavfile as wav

from tempfile import TemporaryFile
import os
import pickle
import random 
import operator
import math
import numpy as np
import pandas as pd
import numpy.matlib
import matplotlib.pyplot as plt
from scipy.interpolate import make_interp_spline

#Extract features from audio wav files using mfcc function in python_speech_features.
def extractFeatures(file_path, target_file, i):
    #samplerate: the samplerate of the signal we are working with.
    #signal: the audio signal from which to compute features.
    samplerate,signal = wav.read(file_path)
    #each 30-sec audio file contains 661794 signals. 
    #winlen: the length of the analysis window in seconds. Set it to 0.020 seconds.
    #the number of dimensions of each mfcc is defaulted to 13. And each 30-second audio has 2994 frames.
    #so, the shape of mfcc_feat of each file is (2994,13).
    mfcc_feat = mfcc(signal, samplerate, winlen=0.020, appendEnergy = False)
    #shape of mean_matrix is 13*1.
    mean_matrix = mfcc_feat.mean(0)
    feature = (mean_matrix, i)  
    pickle.dump(feature , target_file)

#Extract the audio feattures from all genres in the dataset, and save features into a dat file.
filepath = "D:/study/coen240/homework/dataset"
f= open("my.dat" ,'wb')
i=0
number_genre = 10
for folder_list in os.listdir(filepath):
    i+=1
    print(i)
    if i==number_genre+1:
        break
    j=0
    for files in os.listdir(filepath+"/"+folder_list):
        extractFeatures(filepath+"/"+folder_list+"/"+files, f, i)
f.close()

#load data of features from file.
dataset=[]
def loadDataset(filename, split, tr, te):
    with open(filename , 'rb') as f:
        while True:
            try:
                dataset.append(pickle.load(f))
            except EOFError:
                f.close()
                break
#split training set and test set from the dataset.
    for x in range(len(dataset)):
        if random.random()<split:
            tr.append(dataset[x])
        else:
            te.append(dataset[x])
trainingset = []
test = []
loadDataset("my.dat" , 0.9, trainingset, test)
print("train is ",len(trainingset))
print("test is ",len(test))

#Apply Euclidean distance.
def euc_distance(instance1, instance2):
    distance = 0
    mm1 = instance1[0]
    mm2 = instance2[0]
    diff = mm1-mm2
    distance = np.dot(diff.T,diff)
    return distance

#use Euclidean distance to find neighbors.
def FindNeighbors(trainingSet, new_audio, k):
    distances = []
    for x in range (len(trainingSet)):
        dist = euc_distance(trainingSet[x], new_audio)
        distances.append((trainingSet[x][1],dist))
    distances.sort(key=operator.itemgetter(1))
    neighbors = []
    for x in range(k):
        neighbors.append(distances[x][0])
    return neighbors

#predict the genres.
def PredictGenre(neighbors):
    classVote = {}
    for x in range(len(neighbors)):
        response = neighbors[x]
        if response in classVote:
            classVote[response]+=1 
        else:
            classVote[response]=1

    sorter = sorted(classVote.items(), key = operator.itemgetter(1), reverse=True)
    return sorter[0][0]

#Try different K values.
K_value = (4, 6, 8, 10, 12)
p = []
for j in range(len(K_value)):
    k = K_value[j]
    print("k is ",k)
    neighbor = []
    for i in range(len(test)):
        neighbor.append(FindNeighbors(trainingset, test[i], k))
    predict_genre = []
    for i in range(len(neighbor)):
        if k==8:
            p.append(PredictGenre(neighbor[i]))
        predict_genre.append(PredictGenre(neighbor[i]))
    acc = 0
    for i in range(len(test)):
        print("Predicted genre is: ",predict_genre[i]," ,the true genre is: ",test[i][1])
        if predict_genre[i]==test[i][1]:
            acc += 1
    print("Accuracy is ",'percent: {:.2%}'.format(acc/len(test)))
#Calculate accuracy for each genre.
acc_genre = np.zeros((number_genre,1))
count = np.zeros((number_genre,1))
for i in range(len(test)):
    count[test[i][1]-1] += 1
    if(p[i]==test[i][1]):
        acc_genre[test[i][1]-1]+=1
x=[]
y=[]

#Plot accuracy for each genre.
for i in range(number_genre):
    x.append(i+1)
    y.append(acc_genre[i]/count[i])
x_new = np.linspace(1,10,1000)
model = make_interp_spline(x,y)
ys = model(x_new)
plt.plot(x, y, c='red')
plt.title("KNN algorithm")
plt.legend(title='K=8')
plt.xlabel("# Genre")
plt.ylabel("Accuracy")
plt.show()