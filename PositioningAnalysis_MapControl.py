# -*- coding: utf-8 -*-
"""
Created on  02 june  1 10:36:11 2024

Project: League of Legend Data Analysis

# finding different zones in map of LOL
@author: Fazilat
"""
import csv
import shutil
import glob
import os

from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn import mixture
from sklearn.preprocessing import StandardScaler

from sklearn.pipeline import Pipeline

from sklearn.impute import SimpleImputer
from kneed import KneeLocator
import imageio
import pyvista as pv
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

os.environ["OMP_NUM_THREADS"] = '4'
def visulize_matchonMap(df,plt):

    for i in range(0,10):
        posx=str(i+1)+'_position_x'
        posy=str(i+1)+'_position_y'

        x = df[posx]
        y = df[posy]


            # Add the background image

        plt.scatter(x, y, c='blue')#c=color_list[i])

        plt.xlabel('X-axis')
        plt.ylabel('Y-axis')
        plt.title(f'Scatter Plot for ')

        # Display the plot
    #plt.show()

color_list = ['red', 'green', 'blue', 'yellow', 'orange', 'purple', 'pink', 'brown', 'gray', 'cyan']

files_limit=50
inputdata_path = "../RiotProject/data/OutputMatchTimeline1/*.csv"
files_processed=0
csv_files = glob.glob(inputdata_path)

for csv_file in csv_files:
    # Read the CSV file into a DataFrame
    print (csv_file, files_processed)
    df = pd.read_csv(csv_file)
    matchId = df.iloc[0, -1]
    df.drop('matchId', axis=1, inplace=True)
    df = df.iloc[:, 1:]
    files_processed=files_processed+1
    print(files_processed)
    if files_processed > files_limit:
         break
    if files_processed == 1:
        x=15000
        y=15000
        extent=0,15000,0,15000
       # extent = np.min(x), np.max(x), np.min(y), np.max(y)
        background_image = plt.imread('LOLmap.jpg')
        plt.imshow(background_image, extent=extent, aspect='auto')

    visulize_matchonMap(df,plt)
plt.show()
