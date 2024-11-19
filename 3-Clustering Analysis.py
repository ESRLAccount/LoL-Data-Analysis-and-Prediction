# -*- coding: utf-8 -*-
"""
Created on  21 May  1 10:36:11 2024

Project: League of Legend Data Analysis

# applying clustring algorithm to find number of cluster (elbow, silhouette), and based on the optimmum
number of cluster, we do clustring the data, nad save the result in folder ClusterResults for each game phase
density plot
heatmap density plot
difference density plot
@author: Fazilat
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import plotly.graph_objs as go
from plotly.subplots import make_subplots
from sklearn.decomposition import PCA
from datetime import datetime
import hdbscan
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler

from sklearn.metrics import davies_bouldin_score
from sklearn.cluster import DBSCAN

from scipy.interpolate import griddata
from scipy.ndimage import gaussian_filter
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from PIL import Image
import base64
from io import BytesIO
import csv
from scipy.stats import gaussian_kde
import shutil
# sklearn


os.environ["OMP_NUM_THREADS"] = '4'
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.manifold import TSNE
from collections import Counter
import seaborn as sns

from sklearn import mixture
from sklearn.preprocessing import StandardScaler

from sklearn.pipeline import Pipeline

from sklearn.impute import SimpleImputer
from kneed import KneeLocator
from kypy import cluster_range, intra_to_inter, plot_internal, plot_clusters
import warnings
warnings.filterwarnings("ignore")


def Visulalize_ZoneonMap(filename):

    df = pd.read_csv(filename)  # ,nrows=10000)

    # Create a color map for the clusters
    colors = ['r', 'g', 'b', 'c', 'm', 'y', 'k', 'orange', 'purple', 'brown', 'magenta']
    i=0
    df=df[df['lane']==('JUNGLE')]
    # Plot the scatter plot for each cluster
    for cls in df['lane'].unique():
        cluster_data = df[df['lane'] == cls]

        plt.scatter(cluster_data['position_x'], cluster_data['position_y'], color=colors[i],
                    label=f'Cluster {cls}')
        i=i+1
    # Add labels and title
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title('Scatter Plot for Clusters')
    plt.legend()

    # Show the plot
    plt.show()

def visualize_zonalMapperplayerRank(filename):
    df = pd.read_csv(filename)  # ,nrows=10000)
    df['individualPosition'] = df['individualPosition'].replace('NONE', 'JUNGLE')
    df['lane'] = df['lane'].replace('NONE', 'JUNGLE')
    cluster_data = df[df['tier'] == 'CHALLENGER']
    i=0
    for cls in cluster_data['lane'].unique():
         df_cluster = cluster_data[cluster_data['lane'] == cls]
         Visualize_LolMapBasedonPosition(df_cluster,cls,i)
         i=i+1

def assign_PhasetoPositionrowData(df):
# Function to assign values to 'P'
    def assign_p(rank):
        if rank < 10:
            return 1
        elif rank < 20:
            return 2
        else:
            return 3

    # Group by columns 'A' and 'B', then apply the function to assign 'P'
    df['Phase'] = df.groupby(['matchId', 'participantId']).cumcount().apply(assign_p)
    return (df)

def filter_2024MatchTimelineData(df):
    # Filter rows where the year is '2023'
    rows_starting_with_2024 = df[df['matchId'] > 'EUW1_6745998743']

    # Get the number of such rows
    num_rows = len(rows_starting_with_2024)

    print(num_rows)

    return(rows_starting_with_2024)

def DifferenceHeatmap(filename):

    df_all = pd.read_csv(filename)
    # df_all['individualPosition'] = df_all['individualPosition'].replace('NONE', 'JUNGLE')
    # df_all['time'] = df_all.groupby(['matchId', 'participantId']).cumcount() + 1
    # df_all.to_csv(outputdata_pathtimeline,index=False)

    df_all = filter_2024MatchTimelineData(df_all)

    df_all = df_all[df_all['participantId'] < 6]
    # Determine grid size
    rows = 3
    cols = 5

    # Create a figure with subplots
    fig, axes = plt.subplots(rows, cols, figsize=(40, 16))

    # Flatten the axes array for easy iteration
    axes = axes.flatten()
    i=-1

    for phase in df_all['Phase'].unique():
        df_phase = df_all[df_all['Phase'] == phase]
        for cls in Role_list:
            print (cls)
            i=i+1
            ax = axes[i]
            df=df_phase[df_phase['individualPosition']==cls]
            make_DensityHeatmapDiff(df, ax, phase, cls)
    plt.show()

def make_DensityHeatmapDiffRanks(df,ax,rank1,rank2,phase,cls):
    # Separate data for w=0 and w=1
    background_image = plt.imread('../Real_LOLMap.png')
    df_w0 = df[df['tier'] == rank1]
    df_w1 = df[df['tier'] == rank2]

    """    if len(df_w0)>10000:
       df_w0 = df_w0.sample(n=10000, random_state=1)
    if len(df_w1)>10000:
       df_w1 = df_w1.sample(n=10000, random_state=1)
    """
    # Set up grid for heatmap
    extent = 0, 15000, 0, 15000
    ax.imshow(background_image, extent=extent, aspect='auto')
    xmin, xmax, ymin, ymax = 0, 15000, 0, 15000

    xgrid, ygrid = np.mgrid[xmin:xmax:100j, ymin:ymax:100j]
    positions = np.vstack([xgrid.ravel(), ygrid.ravel()])

    ax.set_xlim(0, 15000)
    ax.set_ylim(0, 15000)

    # Compute kernel density estimates for w=0 and w=1
    kde_w0 = gaussian_kde(np.vstack([df_w0['position_x'], df_w0['position_y']]))
    kde_w1 = gaussian_kde(np.vstack([df_w1['position_x'], df_w1['position_y']]))

    # Evaluate densities on the grid
    density_w0 = np.reshape(kde_w0(positions).T, xgrid.shape)
    density_w1 = np.reshape(kde_w1(positions).T, xgrid.shape)

    # Calculate the difference
    density_diff = density_w1 - density_w0


    # Normalize the density difference
    max_diff = np.max(np.abs(density_diff))  # Get the maximum absolute difference
    density_diff_normalized = density_diff / max_diff  # Normalize between -1 and 1

    # Apply thresholding to highlight significant differences
    threshold = 0.1  # Adjust threshold for highlighting significant differences
    density_diff_thresholded = np.where(np.abs(density_diff_normalized) > threshold,
                                        density_diff_normalized, 0)

    # Plot using contourf with a custom colormap and levels

    cmap = plt.get_cmap('coolwarm')
    contour = ax.contourf(xgrid, ygrid, density_diff_thresholded, levels=100, cmap=cmap, alpha=0.6,vmin=-1, vmax=1)

    # Add a color bar
    plt.colorbar(contour, ax=ax)


    #ax.contourf(xgrid, ygrid, density_diff, levels=50, cmap='coolwarm', fill=None, thresh=0.1, alpha=0.4)


    # contour=ax.contourf(xgrid, ygrid, density_diff, levels=100, cmap='seismic', fill=True, vmin=-1, vmax=1, alpha=0.8)

    #ax.set_title(f'role={cls} phase={phase} w=1 vs w=0')

def make_DensityHeatmapDiff(df,ax,phase,cls):
    # Separate data for w=0 and w=1
    background_image = plt.imread('../RealLOLMap.png')
    #if len(df) > 50000:
    #    df = df.sample(n=50000, random_state=42)

    df_w0 = df[df['win'] == 0]
    df_w1 = df[df['win'] == 1]

    """    if len(df_w0)>10000:
       df_w0 = df_w0.sample(n=10000, random_state=1)
    if len(df_w1)>10000:
       df_w1 = df_w1.sample(n=10000, random_state=1)
    """
    # Set up grid for heatmap
    extent = 0, 16000, 0, 16000
    ax.imshow(background_image, extent=extent, aspect='auto')
    xmin, xmax, ymin, ymax = 0, 16000, 0, 16000

    xgrid, ygrid = np.mgrid[xmin:xmax:100j, ymin:ymax:100j]
    positions = np.vstack([xgrid.ravel(), ygrid.ravel()])

    ax.set_xlim(0, 16000)
    ax.set_ylim(0, 16000)

    # Compute kernel density estimates for w=0 and w=1
    kde_w0 = gaussian_kde(np.vstack([df_w0['position_x'], df_w0['position_y']]))
    kde_w1 = gaussian_kde(np.vstack([df_w1['position_x'], df_w1['position_y']]))

    # Evaluate densities on the grid
    density_w0 = np.reshape(kde_w0(positions).T, xgrid.shape)
    density_w1 = np.reshape(kde_w1(positions).T, xgrid.shape)

    # Calculate the difference
    density_diff = density_w1 - density_w0

    # Normalize the density difference
    max_diff = np.max(np.abs(density_diff))  # Get the maximum absolute difference
    density_diff_normalized = density_diff / max_diff  # Normalize between -1 and 1

    # Apply thresholding to highlight significant differences
    threshold = 0.1  # Adjust threshold for highlighting significant differences
    density_diff_thresholded = np.where(np.abs(density_diff_normalized) > threshold,
                                        density_diff_normalized, 0)

    # Plot using contourf with a custom colormap and levels

    cmap = plt.get_cmap('coolwarm')
    #contour = ax.contourf(xgrid, ygrid, density_diff_thresholded, levels=100, cmap=cmap, alpha=0.6,vmin=-1, vmax=1)

    # Add a color bar
    #plt.colorbar(contour, ax=ax)

    contour_filled = ax.contourf(xgrid, ygrid, density_diff_thresholded, levels=100, cmap=cmap, alpha=0.4, vmin=-1,
                                 vmax=1)

    # Add contour lines to show the borders between the zones
    contour_lines = ax.contour(xgrid, ygrid, density_diff_thresholded, levels=100, colors='white', linewidths=0.3)

    # Add a color bar for the filled contour plot
    plt.colorbar(contour_filled, ax=ax)

    # Optionally, label the contour lines (can be useful to highlight values on the borders)
    ax.clabel(contour_lines, inline=True, fontsize=8, fmt="%.2f")

    #ax.contourf(xgrid, ygrid, density_diff, levels=50, cmap='coolwarm', fill=True, thresh=0, alpha=0.4)

    #sns.scatterplot(x='position_x', y='position_y', data=df, s=1, color='black', alpha=0.5,ax=ax)
    #sns.kdeplot(x='position_x', y='position_y', data=df, cmap='coolwarm', fill=None, thresh=0, levels=100,
    #            alpha=0.5,ax=ax)




def ST_DBSCANClustering(filename):

  df_all = pd.read_csv(filename )
  #df_all['individualPosition'] = df_all['individualPosition'].replace('NONE', 'JUNGLE')
  #df_all['time'] = df_all.groupby(['matchId', 'participantId']).cumcount() + 1
  #df_all.to_csv(outputdata_pathtimeline,index=False)

  #df_all=filter_2024MatchTimelineData(df_all)

  df_all=df_all[df_all['participantId'] <6]

  #Make_densityheatmap_PerPhase(df_all)
  #spatio_temporalHeatmap(df_all)

  #Make_densityheatmap_RolesPerWin(df_all)

  #Make_densityheatmap_Roles(df_all)

  Make_densityheatmap_Ranks(df_all)

  #Make_densityheatmap_DiffTwoRanks(df_all)

def Make_densityheatmap_DiffTwoRanks(df_all):

    rankcompare_list=['GOLD','IRON']
    rows = 3
    cols = len(rankcompare_list)

    # Create a figure with subplots
    fig, axes = plt.subplots(rows, cols, figsize=(15, 9))

    # Flatten the axes array for easy iteration
    axes = axes.flatten()
    role = 'JUNGLE'
    df_all=df_all[df_all['individualPosition']==role]
    i = 0
    for cls in rankcompare_list:
        for phase in df_all['Phase'].unique():

            df_phase = df_all[df_all['Phase'] == phase]

            make_DensityHeatmapDiffRanks(df_phase, axes[i], cls, 'CHALLENGER', phase, 'diff')
            i=i+1
    plt.tight_layout()
    plt.show()

def Make_densityheatmap_Ranks(df_all):
        #win = 1
        #df_all = df_all[df_all['win'] == win]
        # rank='EMERALD'
        # df_all=df_all[df_all['tier']==rank]
        # Determine grid size
        rows = 3
        cols = len(Rank_list)

        # Create a figure with subplots
        fig, axes = plt.subplots(rows, cols, figsize=(18, 12))
        # Flatten the axes array for easy iteration
        axes = axes.flatten()
        """ fig.patch.set_alpha(0)
        for ax in axes:
            ax.patch.set_alpha(0)  # Makes each subplot background transparent
            # Remove all borders, ticks, and axis labels
            ax.set_xticks([])
            ax.set_yticks([])
            ax.spines['top'].set_visible(False)
            ax.spines['bottom'].set_visible(False)
            ax.spines['left'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.set_xticklabels([])
            ax.set_yticklabels([])
            ax.set_title("", fontsize=0)  # No title

        """
        role = 'JUNGLE'
        df_all = df_all[df_all['individualPosition'] == role]
        i = -1
        for phase in df_all['Phase'].unique():
            #df_phase=df_all
            df_phase = df_all[df_all['Phase'] == phase]
            for cls in Rank_list:
                i = i + 1
                print(cls)

                df = df_phase[df_phase['tier'] == cls]
                print(len(df))
                if len(df) > 6000:
                    df = df.sample(n=6000, random_state=1)

                # Normalize spatial and temporal features
                # scaler = MinMaxScaler()
                # df[['position_x', 'position_y']] = scaler.fit_transform(df[['position_x', 'position_y']])
                # df['time'] = scaler.fit_transform(df[['time']])
                # print(df['time'])

                features = df[['position_x', 'position_y', 'time']].values

                # Initialize DBSCAN with modified parameters
                eps_value = 0.3  # Increase this value to allow more points in the same cluster
                min_samples_value = 5  # Decrease this value to allow clusters to form more easily

                # clusterer = DBSCAN(eps=eps_value, min_samples=min_samples_value,metric='euclidean')
                """
                clusterer = hdbscan.HDBSCAN(min_cluster_size=10, metric='euclidean')
                df['cluster_label'] = clusterer.fit_predict(features)
                no_lables = len(df['cluster_label'].unique())

                if no_lables > 1:
                    db_index = davies_bouldin_score(features, df['cluster_label'])
                    print(f'Davies-Bouldin Index: {db_index} for phase= {phase} and {cls}')
                    df = df[df['cluster_label'] != -1]
                    # df.to_csv(output_filename6)
                    #spatio_temporalHeatmap(df)

                    plot_density_with_sequence(df, 6, win, cls, phase, role, axes[i])
                """
                Visualize_LolMapBasedonPosition_ax(df, 'all', cls, phase, axes[i])

                # visulize_spacecubWithHeatmap(df,phase,cls)
                # cluster_description(idf)
                # cluster_densityAnalysis(df)
                # Cluster_Adjacency(df)
                # Temporal_analysis(df)
                # desity_overlap(df)
                # Call the function to plot temporal patterns
                # plot_cluster_activity_over_time(df,10,win)

                # Call the function to analyze temporal sequence of clusters
                # analyze_cluster_sequence(df)
                # Remove empty subplots if any
        # for j in range(i + 1, len(axes)):
        #  fig.delaxes(axes[j])

        # Adjust layout and show all plots at once
        #plt.tight_layout()
        plt.show()

def Make_densityheatmap_RolesPerWin(df_all):


    # rank='EMERALD'
    # df_all=df_all[df_all['tier']==rank]
    rank = 'all'
    # Determine grid size
    rows = 3
    cols = 3
    win_list=[0,1]
    # Create a figure with subplots
    fig, axes = plt.subplots(rows, cols, figsize=(24,21))
    role='JUNGLE'
    df_all = df_all[df_all['individualPosition']==role]
    # Flatten the axes array for easy iteration
    axes = axes.flatten()
    i = -1
    for phase in df_all['Phase'].unique():

        df_phase = df_all[df_all['Phase'] == phase]
        for cls in win_list:
            i = i + 1
            print(cls)

            df = df_phase[df_phase['win'] == cls]
            df.to_csv(f'{cls}.csv', index=False)
            print(len(df))
            #if len(df) > 50000:
            #    df = df.sample(n=50000, random_state=42)
            #Visualize_LolMapBasedonPosition_ax(df,role,cls,phase,axes[i])
            #plt.show()
        ###adding diffrence heatmap
        i=i+1
        make_DensityHeatmapDiff(df_phase, axes[i], phase, 'diff')
        #plt.show()
    plt.tight_layout()
    #plt.subplots_adjust(wspace=0.1, hspace=0.1)
    plt.show()

def Make_densityheatmap_PerPhase(df_all):
    rows = 1
    cols = 3
    # Create a figure with subplots
    fig, axes = plt.subplots(rows, cols, figsize=(27,9))
    #role='JUNGLE'
    #df_all = df_all[df_all['individualPosition']==role]
    # Flatten the axes array for easy iteration
    axes = axes.flatten()
    fig.patch.set_alpha(0)
    for ax in axes:
        ax.patch.set_alpha(0)  # Makes each subplot background transparent
        # Remove all borders, ticks, and axis labels
        ax.set_xticks([])
        ax.set_yticks([])
        ax.spines['top'].set_visible(False)
        ax.spines['bottom'].set_visible(False)
        ax.spines['left'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_title("", fontsize=0)  # No title

    i = -1
    for phase in df_all['Phase'].unique():
        df_phase = df_all[df_all['Phase'] == phase]
        i = i + 1
        print(len(df_phase))
        #if len(df) > 50000:
        #    df = df.sample(n=50000, random_state=42)
        Visualize_LolMapBasedonPosition_ax(df_phase,'all','winloss',phase,axes[i])
        #plt.show()


    plt.tight_layout()
    plt.show()

def Make_densityheatmap_Roles(df_all):

  win = 0
  # df_all = df_all[df_all['win'] == win]
  # rank='EMERALD'
  # df_all=df_all[df_all['tier']==rank]
  rank = 'all'
  # Determine grid size
  rows = 3
  cols = 10

  # Create a figure with subplots
  fig, axes = plt.subplots(rows, cols, figsize=(20, 8))

  # Flatten the axes array for easy iteration
  axes = axes.flatten()
  i = -1
  for phase in df_all['Phase'].unique():

      df_phase = df_all[df_all['Phase'] == phase]
      for cls in Role_list:
          i = i + 1
          print(cls)

          df = df_phase[df_phase['individualPosition'] == cls]
          print(len(df))
          if len(df) > 100000:
              df = df.sample(n=100000, random_state=1)

          # Normalize spatial and temporal features
          # scaler = MinMaxScaler()
          # df[['position_x', 'position_y']] = scaler.fit_transform(df[['position_x', 'position_y']])
          # df['time'] = scaler.fit_transform(df[['time']])
          # print(df['time'])

          features = df[['position_x', 'position_y', 'time']].values

          # Initialize DBSCAN with modified parameters
          eps_value = 0.3  # Increase this value to allow more points in the same cluster
          min_samples_value = 5  # Decrease this value to allow clusters to form more easily

          # clusterer = DBSCAN(eps=eps_value, min_samples=min_samples_value,metric='euclidean')

          clusterer = hdbscan.HDBSCAN(min_cluster_size=200, metric='euclidean')
          df['cluster_label'] = clusterer.fit_predict(features)
          no_lables = len(df['cluster_label'].unique())

          if no_lables > 1:
              db_index = davies_bouldin_score(features, df['cluster_label'])
              print(f'Davies-Bouldin Index: {db_index} for phase= {phase} and {cls}')
              df = df[df['cluster_label'] != -1]
              # df.to_csv(output_filename6)

              #plot_density_with_sequence(df, 6, win, cls, phase, rank, axes[i])
          visulize_spacecubWithHeatmap(df,phase)
          # cluster_description(df)
          # cluster_densityAnalysis(df)
          # Cluster_Adjacency(df)
          # Temporal_analysis(df)
          # desity_overlap(df)
          # Call the function to plot temporal patterns
          # plot_cluster_activity_over_time(df,10,win)

          # Call the function to analyze temporal sequence of clusters
          # analyze_cluster_sequence(df)
          # Remove empty subplots if any
  # for j in range(i + 1, len(axes)):
  #  fig.delaxes(axes[j])

  # Adjust layout and show all plots at once
  plt.tight_layout()
  plt.show()

def cluster_densityAnalysis(df):
    # Area of each cluster's bounding box
    cluster_areas = df.groupby('cluster_label').apply(lambda g: (g['position_x'].max() - g['position_x'].min()) * (g['position_y'].max() - g['position_y'].min()))

    cluster_sizes = df['cluster_label'].value_counts().sort_index()
    # Density of each cluster
    cluster_densities = cluster_sizes / cluster_areas

    print("Cluster Densities (Number of Points per Unit Area):")
    print(cluster_densities)

    # Plotting cluster densities
    plt.figure(figsize=(10, 6))
    cluster_densities.plot(kind='bar', color='lightgreen')
    plt.title("Cluster Densities")
    plt.xlabel("Cluster Label")
    plt.ylabel("Density (Points per Unit Area)")
    plt.show()

    descriptive_stats = df.groupby('cluster_label').agg({
        'position_x': ['mean', 'std'],
        'position_y': ['mean', 'std'],
        'time': ['mean', 'std']
    })

    print("Descriptive Statistics for Each Cluster:")
    print(descriptive_stats)

def Cluster_Adjacency(df):
    from scipy.spatial.distance import cdist


    # Assuming df contains your data with 'x', 'y', and 'cluster' columns

    # Calculate centroids of each cluster
    centroids = df.groupby('cluster_label')[['position_x', 'position_y']].mean()

    # Compute the pairwise distances between cluster centroids
    distances = cdist(centroids, centroids)

    # Convert to a DataFrame for easier analysis
    distance_df = pd.DataFrame(distances, index=centroids.index, columns=centroids.index)

    print("Pairwise cluster centroid distances:")
    print(distance_df)

    # Plot a heatmap of distances
    plt.figure(figsize=(10, 8))
    plt.title("Cluster Centroid Distance Matrix")
    sns.heatmap(distance_df, annot=True, cmap='coolwarm')
    plt.show()

def cluster_description(df):
    # Assuming df is your DataFrame with 'cluster' column
    cluster_sizes = df['cluster_label'].value_counts().sort_index()

    print("Cluster Sizes (Number of Points in Each Cluster):")
    print(cluster_sizes)

    # Plotting the sizes of clusters
    plt.figure(figsize=(10, 6))
    cluster_sizes.plot(kind='bar', color='skyblue')
    plt.title("Cluster Sizes")
    plt.xlabel("Cluster Label")
    plt.ylabel("Number of Points")
    plt.show()
def Temporal_analysis(df):
    # Find the first appearance of each cluster
    first_appearance = df.groupby('cluster_label')['time'].mean().sort_values()

    print("Order of cluster appearances over time:")
    print(first_appearance)

    # Plot temporal sequence of cluster appearances
    plt.figure(figsize=(12, 6))
    plt.plot(first_appearance.values, marker='o', linestyle='-', color='blue')
    plt.title("Temporal Sequence of Cluster Appearances")
    plt.xlabel("Cluster")
    plt.ylabel("Time of First Appearance")
    plt.grid(True)
    plt.show()

def desity_overlap(df):

    cluster_1 = df[df['cluster_label'] == 1]
    cluster_2 = df[df['cluster_label'] == 2]

    plt.figure(figsize=(10, 8))
    # Plot density for cluster 1
    sns.kdeplot(x='position_x', y='position_y', data=cluster_1, cmap='Blues', fill=None, thresh=0, levels=100, alpha=0.5)

    sns.kdeplot(x='position_x', y='position_y', data=cluster_1, cmap='Reds', fill=None, thresh=0, levels=100, alpha=0.5)


    plt.legend()
    plt.title("Density Overlap Between Clusters 1 and 2")
    plt.show()
##Temporal Patterns within Clusters
def plot_cluster_activity_over_time(df, top_n,win):
    top_clusters = df['cluster_label'].value_counts().nlargest(top_n).index

    # Filter the DataFrame to only include the top N clusters
    df_top_clusters = df[df['cluster_label'].isin(top_clusters)]

    # Create a new label that combines x and y positions
    df_top_clusters['xy_label'] = df_top_clusters.apply(lambda row: f"({row['position_x']}, {row['position_y']})", axis=1)

    # Group by the new label and time to get counts
    cluster_activity = df_top_clusters.groupby(['xy_label', 'time']).size().unstack(fill_value=0)

    # Plotting the activity over time
    plt.figure(figsize=(12, 8))
    for label in cluster_activity.index:
        plt.plot(cluster_activity.columns, cluster_activity.loc[label], marker='o', label=label)

    # Add labels and title
    plt.title(f'Cluster Activity Over Time for Top {win} Clusters')
    plt.xlabel('Time')
    plt.ylabel('Number of Points')
    plt.legend(title='(X, Y) Position')
    plt.grid(True)
    plt.show()


###Temporal Sequence of Cluster Appearance
def analyze_cluster_sequence(df):
    # Sort the dataframe by time
    df_sorted = df.sort_values(by='time')

    # Identify the first appearance of each cluster
    first_appearance = df_sorted.groupby('cluster_label')['time'].min().reset_index()

    # Sort clusters by their first appearance
    first_appearance_sorted = first_appearance.sort_values(by='time')

    # Print the sequence of cluster appearances
    print("Temporal Sequence of Cluster Appearance:")
    print(first_appearance_sorted)
    #plot_cluster_sequence(df)


def plot_density_with_sequence(df, top_n,win,role,phase,rank,ax):
    # Identify the top N most populous clusters
    top_clusters = df['cluster_label'].value_counts().nlargest(top_n).index

    # Filter the DataFrame to only include the top N clusters
    #df = df[df['cluster_label'].isin(top_clusters)]

    # Identify the first appearance of each cluster
    first_appearance = df.sort_values(by='time').groupby('cluster_label')['time'].min().reset_index()

    first_appearance = df.sort_values(by='time').groupby('cluster_label').first().reset_index()
    # Filter the first appearance to include only the top N clusters
  #  first_appearance = first_appearance[first_appearance['cluster_label'].isin(top_clusters)]

    # Sort the clusters by their first appearance time
    first_appearance_sorted = first_appearance.sort_values(by='time')

    # Add a sequence label based on the appearance order
    first_appearance_sorted['sequence'] = range(1, len(first_appearance_sorted) + 1)
    # Plot density for top clusters

    background_image = plt.imread('../Real_LOLMap.png')

    extent=0,15000,0,15000


    # Add the background image
    ax.imshow(background_image, extent=extent, aspect='auto')
    ax.set_xlim(0, 15000)
    ax.set_ylim(0, 15000)
    xmin, xmax, ymin, ymax = 0, 15000, 0, 15000

    xgrid, ygrid = np.mgrid[xmin:xmax:100j, ymin:ymax:100j]
    positions = np.vstack([xgrid.ravel(), ygrid.ravel()])

    #sns.scatterplot(x='position_x', y='position_y', data=df, s=1, color='black', alpha=0.5)
    #plt.scatter(df['position_x'], df['position_y'], s=1, color='black', alpha=0.3)
    # Draw the density heatmap
    sns.kdeplot(x='position_x', y='position_y', data=df, cmap='coolwarm', fill=None, thresh=0, levels=100, alpha=0.6,ax=ax)


   # plt.figure(figsize=(12, 8))
   # sns.kdeplot(data=df_top_clusters, x='position_x', y='position_y', hue='cluster_label', fill=None, common_norm=False, palette='viridis',
   #             alpha=0.6)

    # Overlay the sequence of cluster appearance
    #plt.plot(first_appearance_sorted['position_x'], first_appearance_sorted['position_y'], linestyle='-', marker='o', color='red',
    #        markersize=8, linewidth=2, label='Cluster Appearance Sequence')

    # Add labels to the sequence line
    #for i, row in first_appearance_sorted.iterrows():
    #    plt.text(row['position_x'], row['position_y'], f'{int(row["sequence"])}', color='red', fontsize=12, ha='right', va='bottom')

    # Add labels and title
    #ax.set_title(f'win = {win} for {role}---phase={phase}-- {rank}')


def plot_density_for_top_clusters(df, top_n=10):
    # Identify the top N most populous clusters
    top_clusters = df['cluster_label'].value_counts().nlargest(top_n).index

    # Filter the DataFrame to only include the top N clusters
    df_top_clusters = df[df['cluster_label'].isin(top_clusters)]

    Visualize_LolMapBasedonPosition(df_top_clusters,'all','all')

def plot_cluster_sequence(df,win):
    # Sort the dataframe by time
    df_sorted = df.sort_values(by='time')

    # Identify the first appearance of each cluster
    first_appearance = df_sorted.groupby('cluster_label')['time'].min().reset_index()

    # Sort clusters by their first appearance
    first_appearance_sorted = first_appearance.sort_values(by='time')

    # Plot the temporal sequence of cluster appearance
    plt.figure(figsize=(10, 6))
    plt.scatter(first_appearance_sorted['time'], first_appearance_sorted['cluster_label'], color='blue')
    plt.plot(first_appearance_sorted['time'], first_appearance_sorted['cluster_label'], linestyle='-', marker='o',
             color='blue')

    # Adding labels and title
    plt.xlabel('Time')
    plt.ylabel('Cluster')
    plt.title(f'Temporal Sequence of Cluster Appearance for win={win}')
    plt.grid(True)
    plt.show()

def VisualizeZoneonMap_BasedOnPhase(filename):
    df = pd.read_csv(filename)

    df=filter_2024MatchTimelineData(df)

    df=df[df['participantId'] <6]
    df=df[df['win']==True]
   # new_df=assign_PhasetoPositionrowData(df)

    for tier in df['tier'].unique():
        df_tier = df[df['tier'] == tier]
        df_tier = df[df['tier'] == 'DIAMOND']
        df_tier=df
        for cls in df_tier['Phase'].unique():
             df_cluster = df_tier[df_tier['Phase'] == cls]
             if len (df_cluster) >400000:
                 df_cluster = df_cluster.sample(n=400000, random_state=1)  # Select 1000 random samples

             df_cluster = Visualize_LolMapBasedonPosition(df_cluster,cls,tier)


def ClusterRolepositionforZoneonMap(filename,outputfile):

    df = pd.read_csv(filename)#,nrows=10000)
    df = filter_2024MatchTimelineData(df)
    df['individualPosition'] = df['individualPosition'].replace('NONE', 'JUNGLE')
    df = df[df['participantId'] < 6]
    win = 'all'
    cls='JUNGLE'
    df = df[df['individualPosition'] == 'JUNGLE']

    tier='GOLD'
    df = df[df['tier'] == tier]
    for phase in df['Phase'].unique():
        cluster_data = df[df['Phase'] == phase]
    #Clustering_PositionPerRole(filename, outputfile, outputfile)
        print (len(cluster_data))
        if len(cluster_data) > 1000:
            cluster_data = cluster_data.sample(n=1000, random_state=1)
    #for tier in df['tier'].unique():
    #     cluster_data = df[df['tier'] == tier]
    #     cluster_df=Visualize_LolMapBasedonPosition(cluster_data,cls,tier)
    #cluster_df = ClusterandVisualize_LolMapBasedonPosition(cluster_data, cls)
        Visualize_LolMapBasedonPosition(cluster_data,cls,tier,phase)
    return

def ClusterRoleRankpositionforZoneonMap(filename,outputfile):

    df = pd.read_csv(filename)#,nrows=10000)
    df['individualPosition'] = df['individualPosition'].replace('NONE', 'JUNGLE')


    cluster_data = df[df['individualPosition'] == 'JUNGLE']

    cluster_data=cluster_data[df['Phase']==3]

    #Clustering_PositionPerRole(filename, outputfile, outputfile)

    for cls in df['tier'].unique():
        cluster_df = cluster_data[cluster_data['tier'] == cls]
        cls=cls + '_' + 'mid game'
        if len(cluster_data)>0 :
             cluster_df=Visualize_LolMapBasedonPosition(cluster_df,cls)

    #cluster_df = Visualize_LolMapBasedonPosition(cluster_data, cls)

    return cluster_df




def Visualize_LolMapBasedonPosition(df,type,rank,phase):
    background_image = plt.imread('../RealLOLMap.png')
    x = df['position_x']
    y = df['position_y']

    extent = np.min(x), np.max(x), np.min(y), np.max(y)
    extent=0,16000,0,16000
    plt.xlim(0, 16000)
    plt.ylim(0, 16000)

    # Add the background image
    plt.imshow(background_image, extent=extent, aspect='auto')
    sns.scatterplot(x='position_x', y='position_y', data=df, s=1, color='black', alpha=0.5)
    #plt.scatter(df['position_x'], df['position_y'], s=1, color='black', alpha=0.3)
    # Draw the density heatmap
    sns.kdeplot(x='position_x', y='position_y', data=df, cmap='coolwarm', fill=None, thresh=0, levels=100, alpha=0.5)
    #hb = plt.hexbin(df['position_x'], df['position_y'], gridsize=50, cmap='coolwarm', mincnt=1, alpha=0.6)
    # Add a color bar
    #cb = plt.colorbar(hb, label='Counts')

    plt.xlabel('X-axis')
    plt.ylabel('Y-axis')

    plt.title(f'density heatmap for {rank} in {getname(type)} -- phase={phase} ')

        # Display the plot
    plt.show()

def Visualize_LolMapBasedonPosition_ax(df, type, rank, phase,ax):
    import matplotlib.image as mpimg
    background_image = mpimg.imread('../Real_LOLMap.png')


    x = df['position_x']
    y = df['position_y']
    # Set global font size
    plt.rcParams.update({'font.size': 18})
    extent = 0, 16000, 0, 16000
    ax.set_xlim(0, 16000)
    ax.set_ylim(0, 16000)

    # Add the background image
    #ax.imshow(background_image, extent=[0, background_image.shape[1], 0, background_image.shape[0]], origin='upper')
    ax.imshow(background_image, extent=extent, aspect='auto')
    if len (df)<60000:
        s=0.5
    else:
        s=0.5
    sns.scatterplot(x='position_x', y='position_y', data=df, s=1, color='black', alpha=0.5,ax=ax)
    # plt.scatter(df['position_x'], df['position_y'], s=1, color='black', alpha=0.3)
    # Draw the density heatmap
    sns.kdeplot(x='position_x', y='position_y', data=df, cmap='coolwarm', fill=None, thresh=0, levels=100,
                alpha=0.5,ax=ax,  bw_adjust=s)


    # Customize font sizes
    #ax.set_xlabel('X', fontsize=18,fontweight='bold')
   # ax.set_ylabel('Y', fontsize=18,fontweight='bold')
   # ax.set_title( rank, fontsize=16,fontweight='bold')
    x_ticks = np.arange(0, 16000, 4000)  # Start from 0, end at max(x), step by 4000
    y_ticks =np.arange(0, 16000, 4000)  # Example for y-axis ticks

    ax.set_xticks(x_ticks)
    ax.set_yticks(y_ticks)
    ax.tick_params(axis='both', which='major', labelsize=12)  # Font size for ticks

    #ax.set_title(f'density heatmap for {rank} in {getname(type)} -- phase={phase} ')


def getname(phase):
    if phase==1:
        return 'Early game'
    if phase==2:
        return 'Mid game'
    if phase==3:
        return 'Late game'
    else:
        return phase


def visualize_clustersonMap(df_cluster):

    colors = ['#43AA8B', '#CDB4DB', '#F1FAEE', '#FFAFCC', '#219EBC', 'y', 'k', 'orange', 'purple', 'brown', 'magenta']

    background_image = plt.imread('../LOLmap.jpg')

    x = df_cluster['position_x']
    y = df_cluster['position_y']

    extent = np.min(x), np.max(x), np.min(y), np.max(y)
    idx = 0
    plt.xlim(0, 15000)
    plt.ylim(0, 15000)
    # Set up the matplotlib figure
    plt.figure(figsize=(10, 8))
    # Plot the scatter plot for each cluster
    for cls in df_cluster['cluster_label'].unique():
        cluster_data = df_cluster[df_cluster['cluster_label'] == cls]
        plt.imshow(background_image, extent=extent, aspect='auto')
        ##plt.scatter(cluster_data['position_x'], cluster_data['position_y'], c=colors[idx],s=10)

        # Draw the scatter plot
        sns.scatterplot(x='position_x', y='position_y', data=cluster_data, s=10, color='black', alpha=0.5)

        # Draw the density heatmap
        sns.kdeplot(x='position_x', y='position_y', data=cluster_data, cmap='coolwarm', fill=False, thresh=0)#, levels=20000)

        idx = idx + 1
    # Add labels and title
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title(f'Scatter Plot ')
    plt.legend()

    # Show the plot
    plt.show()
    return (df_cluster)

def ClusterandVisualize_LolMapBasedonPosition(df,tier):
    kmeans_kwargs = {
        "init": "k-means++",
        "n_init": 2,
        "max_iter": 50000,
        "random_state": 42,
    }


    poscols = ['position_x', 'position_y']

    X = df[poscols]


    num_cluster_el = elbow(X, kmeans_kwargs)

    df_cluster = make_clustring(df, X, kmeans_kwargs, num_cluster_el)

    visualize_clustersonMap(df_cluster)



def ClusteringData(filename, output_file,output_file2):
    kmeans_kwargs = {
        "init": "k-means++",
        "n_init": 2,
        "max_iter": 50000,
        "random_state": 42,
    }
    df = pd.read_csv(filename)
    #df1 = df[df['Phase'] == 1]  ###select data related to phase1
    #df.drop_duplicates(inplace=True)
    #filtered_df=splite_dataframe(df1)

    df=df.drop(['participantId','matchId','gameDuration'],axis=1)

    numericCols = df.select_dtypes(include='number').columns

    X=df[numericCols]

    #DNN_Clustering(X)
    # Call elbow algoritm to find number of cluster
    num_cluster_el = elbow(X, kmeans_kwargs)
    #num_cluster_si = silhouette2(X)

    df_cluster=make_clustring(df, X, kmeans_kwargs, num_cluster_el, output_file)
    df_cluster.groupby(['cluster']).mean()
    numeric_columns = [col for col in df.select_dtypes(include='number').columns]
    agg_dict = {col: 'mean' for col in numeric_columns}
    agg_dict['individualPosition'] = 'first'  # Add the first role aggregation
    agg_dict['lane'] = 'first'
    agg_dict['individualPosition'] = 'first'
    agg_dict['win'] = 'first'
    agg_dict['championName'] = 'first'

    # Grouping by 'mid' and 'pid', and aggregating the data
    summary_df = df_cluster.groupby(['cluster']).agg(agg_dict).reset_index()
    summary_df.to_csv(output_file2)
    """
    print('num_cluster for elbow ', num_cluster_el)
    print('num_cluster for silhouette ', num_cluster_si)

        # Call Clustring algoritm to clustr the data and label the data
    if num_cluster_si is not None:

        make_clustring(df1, X, kmeans_kwargs, num_cluster_si, output_file)
    else:
        if num_cluster_el is not None:
            make_clustring(df1, X, kmeans_kwargs, num_cluster_el, output_file)

    return ()
"""


def Clustering_PositionPerRole(filename, output_file,output_file2):
    kmeans_kwargs = {
        "init": "k-means++",
        "n_init": 2,
        "max_iter": 50000,
        "random_state": 42,
    }
    df = pd.read_csv(filename)
    #df1 = df[df['Phase'] == 1]  ###select data related to phase1


    numericCols = ['position_x','position_y']

    X=df[numericCols]

    #DNN_Clustering(X)
    # Call elbow algoritm to find number of cluster
    num_cluster_el = elbow(X, kmeans_kwargs)
    #num_cluster_si = silhouette2(X)

    df_cluster=make_clustring(df1, X, kmeans_kwargs, num_cluster_el, output_file)
    df_cluster.groupby(['cluster']).mean()
    numeric_columns = [col for col in df.select_dtypes(include='number').columns]
    agg_dict = {col: 'mean' for col in numeric_columns}
    agg_dict['individualPosition'] = 'first'  # Add the first role aggregation
    agg_dict['lane'] = 'first'
    agg_dict['individualPosition'] = 'first'
    agg_dict['win'] = 'first'
    agg_dict['championName'] = 'first'

    # Grouping by 'mid' and 'pid', and aggregating the data
    summary_df = df_cluster.groupby(['cluster']).agg(agg_dict).reset_index()
    summary_df.to_csv(output_file2)
    """
    print('num_cluster for elbow ', num_cluster_el)
    print('num_cluster for silhouette ', num_cluster_si)

        # Call Clustring algoritm to clustr the data and label the data
    if num_cluster_si is not None:

        make_clustring(df1, X, kmeans_kwargs, num_cluster_si, output_file)
    else:
        if num_cluster_el is not None:
            make_clustring(df1, X, kmeans_kwargs, num_cluster_el, output_file)

    return ()
"""

# *******************************************************
def silhouette(X, kmeans_kwargs):
    # A list holds the silhouette coefficients for each k
    silhouette_coefficients = []

    num_cluster = int(len(X) - 2)
    max_cluster = 6  # no need to have more than 6 cluster
    n = min(num_cluster, max_cluster)
    max_score = -1
    b_cluster = 0
    if num_cluster >= 3:
        # Notice you start at 2 clusters for silhouette coefficient
        for k in range(2, n):
            gmm = mixture.GaussianMixture(n_components=k)
            labels = gmm.fit_predict(X)

            # kmeans = KMeans(n_clusters=k, **kmeans_kwargs)
            # kmeans.fit(X)
            # score = silhouette_score(X, kmeans.labels_)
            score = silhouette_score(X, labels)
            if max_score < score:
                max_score = score
                b_cluster = k
            silhouette_coefficients.append(score)

        plt.style.use("fivethirtyeight")
        plt.plot(range(2, n), silhouette_coefficients)
        plt.xticks(range(2, n))
        plt.title('The silhouette_coefficients for :' + str(len(X)) )
        plt.xlabel("Number of Clusters")
        plt.ylabel("Silhouette Coefficient")
        plt.show()
        print('Silhouette', b_cluster, max_score)
        return (b_cluster)


# *******************************************************
def elbow(X, kmeans_kwargs):
    cs = []
    num_cluster = int(len(X) - 2)
    max_cluster = 14  # no need to have more than 6 cluster
    n = min(num_cluster, max_cluster)

    if num_cluster >= 3:

        for i in range(1, n):
            kmeans = KMeans(n_clusters=i, **kmeans_kwargs)
            kmeans.fit(X)
            cs.append(kmeans.inertia_)
        plt.plot(range(1, n), cs)
        plt.title('The Elbow Method for :' + str(len(X)) )
        plt.xlabel('Number of clusters')
        plt.ylabel('CS')
        plt.show()
        kl = KneeLocator(range(1, n), cs, curve="convex", direction="decreasing")
        return (kl.elbow)


# *******************************************************
# *******************************************************
def make_clustring(df_original, df, kmeans_kwargs, k):
    #k = 7

    kmeans = KMeans(n_clusters=k, **kmeans_kwargs)
    # X=df['LapTime'].values.reshape(-1,1)
    X = df

    kmeans.fit(X)

    labels = kmeans.fit_predict(X)


    # Adding the results to a new column in the dataframe
    df_original["cluster"] = labels  # kmeans.labels_

    # making readable labels for cluster
    df_original=(Create_clusterLabel(df_original))

    return(df_original)

def get_KPI(inputdata_path2,phase):
    if phase == 0:
        metric_filename = inputdata_path2 + '_all.csv'
    if phase == 1:
        metric_filename = inputdata_path2 + '_Earlygame.csv'
    if phase == 2:
        metric_filename = inputdata_path2 + '_Midgame.csv'
    if phase == 3:
        metric_filename = inputdata_path2 + '_Lategame.csv'

    df=pd.read_csv(metric_filename)
    df=df.sort_values(by=['SHAP Score'],ascending=False)

    #df1=df['Feature']

    metric_list = df.iloc[:10, 0]



    return metric_list
##########################################################

def Create_clusterLabel(df):
    # Determine the point with minimum x and minimum y
    min_point = df.loc[df[['position_x', 'position_y']].sum(axis=1).idxmin()]
    cluster_a = min_point['cluster']

    # Determine the point with maximum x and maximum y
    max_point = df.loc[df[['position_x', 'position_y']].sum(axis=1).idxmax()]
    cluster_b = max_point['cluster']

    # Determine the cluster with points within range (7000, 7000)
    def is_within_range(row,x_range, y_range):
        return (row['position_x'] >= x_range[0]) & (row['position_x'] <= x_range[1]) & \
            (row['position_y'] >= y_range[0]) & (row['position_y'] <= y_range[1])

    x_range = (6000, 7200)
    y_range = (6000, 7200)
    cluster_c = df.loc[df.apply(lambda  row:is_within_range(row,x_range,y_range), axis=1), 'cluster'].iloc[0]
    cluster_d = df.loc[df.apply(lambda  row:is_within_range(row,(0,4000),(11000,14000)), axis=1), 'cluster'].iloc[0]
    cluster_e = df.loc[df.apply(lambda  row:is_within_range(row,(13000,14000),(0,4000)), axis=1), 'cluster'].iloc[0]

    # Assign cluster names
    df['cluster_label'] = df['cluster'].apply(
        lambda cluster: 'A' if cluster == cluster_a else
        'B' if cluster == cluster_b else
        'C' if cluster == cluster_c else
        'D' if cluster == cluster_d else
        'E'
    )
    df.sort_values(by=['cluster_label'], inplace=True)

    return (df)

def assign_PhasetoPositionrowData(df):
# Function to assign values to 'P'
    def assign_p(rank):
        if rank < 10:
            return 1
        elif rank < 20:
            return 2
        else:
            return 3

    # Group by columns 'A' and 'B', then apply the function to assign 'P'
    df['Phase'] = df.groupby(['matchId', 'participantId']).cumcount().apply(assign_p)
    return (df)

#################################################

def visulize_spacecubWithHeatmapdensityXY(df,phase):
    # Extract data
    x = df['position_x']
    y = df['position_y']
    z = df['time']

    # Define the grid ranges with higher resolution
    x_range = np.linspace(1, 15001, 200)  # 200 points in x
    y_range = np.linspace(1, 15001, 200)  # 200 points in y

    # Generate the grid
    #grid_x, grid_y = np.meshgrid(x_range, y_range)

    xmin, xmax, ymin, ymax = 0, 15000, 0, 15000

    grid_x, grid_y = np.mgrid[xmin:xmax:100j, ymin:ymax:100j]

    # Compute point density using a 2D histogram
    density, xedges, yedges = np.histogram2d(x, y, bins=[x_range, y_range])
    density = density.T  # Transpose to match grid dimensions

    # Interpolate z-values (time) over the grid using cubic interpolation
    grid_z = griddata((x, y), z, (grid_x, grid_y), method='cubic')

    # Apply Gaussian filter for smoothing density and time
    sigma_density = 3
    sigma_time = 4
    smoothed_density = gaussian_filter(density, sigma=sigma_density)  # Apply smoothing to density
    smoothed_z = gaussian_filter(grid_z, sigma=sigma_time)  # Apply smoothing to time

    if phase==1:
        smoothed_z = np.clip(smoothed_z, 1, 10)
    if phase==2:
        smoothed_z = np.clip(smoothed_z, 10, 20)
    if phase==3:
        smoothed_z = np.clip(smoothed_z, 20, 40)

    # Normalize smoothed density for color mapping
    normalized_density = (smoothed_density - smoothed_density.min()) / (smoothed_density.max() - smoothed_density.min())

    # Define a custom color scale from blue to red
    density_colorscale = [
        [0, 'rgb(0, 0, 255)'],  # Blue for lowest density
        [1, 'rgb(255, 0, 0)']
    ]

    # Plot the surface with density influencing color and time influencing depth
    fig = go.Figure(data=[go.Surface(
        z=smoothed_z,  # Smoothed interpolated time values
        x=grid_x,  # Position X grid
        y=grid_y,  # Position Y grid
        surfacecolor=normalized_density,  # Color based on density
        colorscale=density_colorscale,  # Choose a color scale
        colorbar=dict(title='Density')  # Add a color bar for reference
    )])

    # Set the layout for the 3D plot with adjusted aspect ratio
    fig.update_layout(
        scene=dict(
            xaxis_title='Position X',
            yaxis_title='Position Y',
            zaxis_title='Time',
            aspectmode='manual',  # Manual adjustment of aspect ratios
            xaxis=dict(
                range=[1, 15001]  # Set the range for the x axis
            ),
            yaxis=dict(
                range=[1, 15001]  # Set the range for the y axis
            ),

            aspectratio=dict(x=1, y=1, z=0.3)  # Adjust the aspect ratio
        ),
        title='3D Surface Plot with Density and Time Visualization',
        margin=dict(l=0, r=0, b=0, t=40)
    )

    # Show the figure
    fig.show()
def visulize_spacecubWithHeatmap(df,phase,type):
    from scipy.interpolate import griddata
    from scipy.ndimage import gaussian_filter
    top_n=10
    # Path to your background image
    top_clusters = df['cluster_label'].value_counts().nlargest(top_n).index

    # Filter the DataFrame to only include the top N clusters
    df = df[df['cluster_label'].isin(top_clusters)]

    # Assuming df is your DataFrame
    x = df['position_x']
    y = df['position_y']
    z = df['time']

    # Define the grid ranges
    x_range = np.arange(1, 15001)
    y_range = np.arange(1, 15001)
    z_range = np.arange(1, 11)

    xmin, xmax, ymin, ymax = 0, 15000, 0, 15000

    grid_x, grid_y = np.mgrid[xmin:xmax:100j, ymin:ymax:100j]
    # Generate the grid
    #x, y, z = np.mgrid[1:15001, 1:15001, 1:11]
    #grid_x, grid_y = np.mgrid[xmin:xmax:grid_size * 1j, ymin:ymax:grid_size * 1j]

    #grid_x, grid_y = np.mgrid[x_range[0]:x_range[1]:15000j, y_range[0]:y_range[1]:15000]  # Increase to 200x200 grid


    #grid_x, grid_y = np.mgrid[xmin:xmax:200j, ymin:ymax:200j]
    #grid_x, grid_y = np.mgrid[xmin:xmax:200j, ymin:ymax:200j]
    # Interpolate z-values (time) over the grid
    grid_z = griddata((x, y), z, (grid_x, grid_y), method='cubic')

    smoothed_z = gaussian_filter(grid_z, sigma=5)  # Adjust sigma for more or less smoothing
    if phase==1:
        smoothed_z = np.clip(smoothed_z, 1, 10)
    if phase==2:
        smoothed_z = np.clip(smoothed_z, 10, 20)
    if phase==3:
        smoothed_z = np.clip(smoothed_z, 20, 40)

    # Define a custom color scale from blue to red
    blue_red_colorscale = [
        [0, 'rgb(0, 0, 139)'],  # Dark blue
        [0.5, 'rgb(255, 255, 255)'],  # Midpoint (white for transition)
        [1, 'rgb(139, 0, 0)']  # Dark red
    ]

    # Plot the surface
    fig = go.Figure(data=[go.Surface(
        z=smoothed_z,  # Interpolated time values
        x=grid_x,  # Position X grid
        y=grid_y,  # Position Y grid
        colorscale=blue_red_colorscale,  # Choose a color scale
        colorbar=dict(title='Time')  # Add a color bar for reference
    )])

    # Set the layout for the 3D plot
    fig.update_layout(
        scene=dict(
            xaxis_title='Position X',
            yaxis_title='Position Y',
            zaxis_title='Time',
            aspectmode='manual',  # Manual adjustment of aspect ratios
            aspectratio=dict(x=1, y=1, z=0.2)  # Adjust the aspect ratio
        ),
        title=f'3D Surface Plot Visualization for phase ={phase} and {type}',
        margin=dict(l=0, r=0, b=0, t=40)
    )

    # Show the figure
    fig.show()
def visulize_spacecub(df):
    top_n=10
    # Path to your background image
    top_clusters = df['cluster_label'].value_counts().nlargest(top_n).index

    # Filter the DataFrame to only include the top N clusters
    df = df[df['cluster_label'].isin(top_clusters)]

    # Create a 3D scatter plot for Space-Time Cube visualization
    fig = go.Figure()
    colors = ['red', 'blue', 'green', 'orange', 'purple', 'cyan', 'magenta', 'yellow', 'brown', 'pink']

    # Create scatter points for each cluster
    for i,cluster in enumerate(df['cluster_label'].unique()):

        cluster_df = df[df['cluster_label'] == cluster]
        fig.add_trace(go.Scatter3d(
            x=cluster_df['position_x'],
            y=cluster_df['position_y'],
            z=cluster_df['time'],
            mode='markers',
            marker=dict(
                size=5,
                color=colors[i % len(colors)],  # Use cluster_label for color
                opacity=0.8
            ),
            name=f'Cluster {cluster}'
        ))

    # Set the layout for the 3D plot
    fig.update_layout(
        scene=dict(
            xaxis_title='Position X',
            yaxis_title='Position Y',
            zaxis_title='Time',
            aspectmode='cube'  # Keep the aspect ratio of the plot cube-like
        ),
        title='Space-Time Cube Visualization',
        margin=dict(l=0, r=0, b=0, t=40)
    )

    # Show the plot
    fig.show()

def spatio_temporalHeatmap(data):

    background_image = plt.imread('../Real_LOLMap.png')
    fig, axes = plt.subplots(2, 5, figsize=(40, 16))
    axes = axes.flatten()
    i=0
    phase=2
    role=data['individualPosition'].unique()
    data=data[data['individualPosition']== 'JUNGLE']
    #data = data[data['tier'] == 'BRONZE']
    data=data[data['Phase']==phase]
    # Create a unique list of time bins
    time_bins = sorted(data['time'].unique())

    # Process each time bin
    for time_bin in time_bins:
      if time_bin<30:
        # Filter data for the current time bin
        ax=axes[i]
        # Filter data for the selected time bin
        filtered_data = data[data['time'] == time_bin]

        if not filtered_data.empty:

            ax.imshow(background_image,
                       extent=[0, 15000, 0, 15000])
            # Overlay the scatter plot
            #sns.scatterplot(data=filtered_data, x='position_x', y='position_y', hue='time', palette='viridis', s=50, edgecolor='w', alpha=0.7,ax=ax)

            sns.scatterplot(x='position_x', y='position_y', data=filtered_data, s=50, color='red', alpha=0.5, ax=ax)

            ax.set_title(f'Spatio-Temporal Heatmap for Time Bin: {time_bin}')


        else:
            print(f"Time bin {time_bin} not found in the dataset.")
        i=i+1
    plt.show()

#### main
print('Start of execution . . . . .')
print('start time: ', datetime.now())



##path for merging three matchtimeline_perPhaseMean to one file
inputdata_path1 = "../data/OutputRank/MatchTimeline/RankedMatchTimeline_withExtraColumns.csv"

inputdata_path2 = "../data/OutputRank/FeatureSelection/KPI_MatchTimelinePerPhase"

outputdata_path = "../data/OutputRank/ClusterResults/"


inputdata_pathtimeline="../data/OutputRank/MatchTimeline/MatchTimeline_masterfile_PositionRowwithRoleRankPhase.csv"
inputdata_pathtimeline_rolePos="../data/OutputRank/ClusterResults/ZoneperMinute_perRole.CSV"
inputdata_pathtimeline_rolePosRank="../data/OutputRank/ClusterResults/ZoneperMinute_perRoleRank.csv"

outputdata_pathtimeline="../data/OutputRank/MatchTimeline/MatchTimeline_masterfile_PositionRowwithRoleRankPhase2024New.csv"
####Creating output directory for the results
##if not os.path.exists(outputdata_path):
##    os.makedirs(outputdata_path)
##else:
    # remove folder with all of its files
##    shutil.rmtree(outputdata_path)

output_filename = outputdata_path + 'CL_MatchTimeline_Phase1.csv'

output_filename2 = outputdata_path + 'CL_MatchTimeline_Phase1Stats.csv'

output_filename3 = outputdata_path + 'CL_MapZone.csv'

output_filename4 = outputdata_path + 'CL_MapZoneNumberofChanged.csv'

output_filename5 = outputdata_path + 'CL_MapZone_role.csv'

output_filename6 = outputdata_path + 'HDBSCANResult.csv'


output_filename7 = outputdata_path + 'HDBSCANResult_TopCluster.csv'

Role_list=['BOTTOM','JUNGLE','TOP','UTILITY','MIDDLE']

Rank_list=['CHALLENGER','EMERALD','BRONZE']




#PCA_data(inputdata_path1,inputdata_path2)
#ClusteringData(inputdata_path1, output_filename,output_filename2)


#Test_InteractiveMapwithTimeDimension()

ST_DBSCANClustering(outputdata_pathtimeline)
#DifferenceHeatmap(outputdata_pathtimeline)
#df_cluster=VisualizeZoneonMap_BasedOnPhase(inputdata_pathtimeline)

#df_cluster=ClusterRolepositionforZoneonMap(inputdata_pathtimeline,output_filename5)

#ClusterRoleRankpositionforZoneonMap(inputdata_pathtimeline_rolePosRank,output_filename5)

#df_cluster.to_csv(output_filename3, index=True)

#df_cluster = find_numberofZoneChanged(df_cluster)
#df_cluster.to_csv(output_filename4, index=True)
#visualize_zonalMapperplayerRank(inputdata_pathtimeline)


#Visulalize_ZoneonMap(inputdata_pathtimeline)