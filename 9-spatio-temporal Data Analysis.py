# -*- coding: utf-8 -*-
"""
Created on  27 May  1 10:36:11 2024

Project: League of Legend Data Analysis

# applying clustring algorithm for spatio temporal analysis to cluster players in each team based on -the location of players
ALPHA SHAP PLOT
SCATER PLOT FOR JUNGLER ROLE
SCATER PLOT FOR RANK PLAYERS
CALCULATION OF TOTAL MOVENET RELATED METRICS

@author: Fazilat

"""
import pandas as pd

import matplotlib.pyplot as plt
import os
import seaborn as sns
from datetime import datetime
import csv
import shutil
import alphashape
from shapely.geometry import Point, Polygon
from scipy.interpolate import splprep, splev
import matplotlib.image as mpimg
import numpy as np

os.environ["OMP_NUM_THREADS"] = '4'
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
#from licpy import lic
import numpy as np
from scipy.interpolate import griddata
from kneed import KneeLocator
from scipy.ndimage import gaussian_filter
from scipy.interpolate import griddata
from scipy.stats import binned_statistic_2d
from scipy.interpolate import make_interp_spline
import imageio
import pyvista as pv

def callculate_movementRelatedScores(input,output):
    df=pd.read_csv(input)
    #result_df=calculate_SplitScore(df,output)
   # calculate_SpliteScorePerPhase(df,output)
    calculate_RotationScorePerPhase(df, output)
    #calculate_CompanionScore(df,output)
    #calculate_RotationScore(df,output)

def calculate_RotationScore(df,output):
    # Ensure data is sorted by matchid, participantId, and time
    df = df.sort_values(by=['matchId', 'participantId', 'time'])

    # Filter for the desired time range (minutes 5 to 20)
    df = df[(df['time'] >= 5) & (df['time'] <= 20)]

    # Calculate distances from origin (0,0) for each point
    df['distance'] = np.sqrt(df['position_x'] ** 2 + df['position_y'] ** 2)

    # Shift distances and positions for the following minute to calculate arc length
    df['distance_next'] = df.groupby(['matchId', 'participantId'])['distance'].shift(-1)
    df['position_x_next'] = df.groupby(['matchId', 'participantId'])['position_x'].shift(-1)
    df['position_y_next'] = df.groupby(['matchId', 'participantId'])['position_y'].shift(-1)

    # Drop rows where next time data is missing (i.e., end of time range for each participant)
    df = df.dropna(subset=['distance_next', 'position_x_next', 'position_y_next'])

    # Calculate the angle between consecutive points
    dot_product = df['position_x'] * df['position_x_next'] + df['position_y'] * df['position_y_next']
    magnitude_product = df['distance'] * df['distance_next']
    df['cos_theta'] = dot_product / magnitude_product
    df['cos_theta'] = df['cos_theta'].clip(-1, 1)  # Avoid numerical errors that lead to cosine values outside [-1, 1]
    df['angle'] = np.arccos(df['cos_theta'])  # Angle in radians

    # Calculate the arc length as average radius * angle
    df['average_radius'] = (df['distance'] + df['distance_next']) / 2
    df['arc_length'] = df['average_radius'] * df['angle']

    # Calculate the Rscore by averaging arc lengths for each participant in each match
    rscore_df = df.groupby(['matchId', 'participantId', 'individualPosition', 'tier','win','championName','lane'])['arc_length'].mean().reset_index()
    rscore_df = rscore_df.rename(columns={'arc_length': 'RotationScore'})

    # Display the final dataframe
    outputfilename=output+'RotationScore.csv'
    rscore_df.to_csv(outputfilename,index=True)

def calculate_RotationScorePerPhase(df,output):
    # Ensure data is sorted by matchid, participantId, and time
    df = df.sort_values(by=['matchId', 'participantId', 'time'])

    # Calculate distances from origin (0,0) for each point
    df['distance'] = np.sqrt(df['position_x'] ** 2 + df['position_y'] ** 2)

    # Shift distances and positions for the following minute to calculate arc length
    df['distance_next'] = df.groupby(['matchId', 'participantId'])['distance'].shift(-1)
    df['position_x_next'] = df.groupby(['matchId', 'participantId'])['position_x'].shift(-1)
    df['position_y_next'] = df.groupby(['matchId', 'participantId'])['position_y'].shift(-1)

    # Drop rows where next time data is missing (i.e., end of time range for each participant)
    df = df.dropna(subset=['distance_next', 'position_x_next', 'position_y_next'])

    # Calculate the angle between consecutive points
    dot_product = df['position_x'] * df['position_x_next'] + df['position_y'] * df['position_y_next']
    magnitude_product = df['distance'] * df['distance_next']
    df['cos_theta'] = dot_product / magnitude_product
    df['cos_theta'] = df['cos_theta'].clip(-1, 1)  # Avoid numerical errors
    df['angle'] = np.arccos(df['cos_theta'])  # Angle in radians

    # Calculate the arc length as average radius * angle
    df['average_radius'] = (df['distance'] + df['distance_next']) / 2
    df['arc_length'] = df['average_radius'] * df['angle']

    # Assign phases based on time thresholds
    conditions = [
        df['time'] < early_phaseTr,
        (df['time'] >= early_phaseTr) & (df['time'] < mid_phaseTr),
        df['time'] >= mid_phaseTr
    ]
    choices = ['1', '2', '3']  # Early Phase = 1, Mid Phase = 2, Late Phase = 3
    df['Phase'] = np.select(conditions, choices, default='1')

    # Group by match, participant, and phase to calculate average arc length
    rscore_df = df.groupby(['matchId', 'participantId', 'Phase', 'individualPosition',
                            'tier', 'win', 'championName', 'lane'])['arc_length'].mean().reset_index()

    # Rename the arc_length column to RotationScore
    rscore_df = rscore_df.rename(columns={'arc_length': 'RotationScore'})

    # Save the final results to a CSV file
    outputfilename = output + 'RotationScore_phases.csv'
    rscore_df.to_csv(outputfilename, index=False)


def calculate_SpliteScorePerPhase(df, output):
    dx = df['position_x'].diff(periods=-1)
    dy = df['position_y'].diff(periods=-1)

    df['xposition_diff'] = dx
    df['yposition_diff'] = dy

    # Calculate the distance using the Pythagorean theorem
    distances = np.sqrt(dx ** 2 + dy ** 2)
    distances.iloc[0] = 0
    df['distance'] = distances

    # Classify the data into phases based on time thresholds
    conditions = [
        df['time'] < early_phaseTr,
        (df['time'] >= early_phaseTr) & (df['time'] < mid_phaseTr),
        df['time'] >= mid_phaseTr
    ]
    choices = ['1', '2', '3']  # 1: Early Phase, 2: Mid Phase, 3: Late Phase
    df['Phase'] = np.select(conditions, choices, default='1')

    # Assign teams based on `participantId`
    df['team'] = df['participantId'].apply(lambda x: 1 if x < 6 else 2)

    # Initialize a list to store the results
    split_scores = []

    # Group by match and team
    for (match_id, team), team_data in df.groupby(['matchId', 'team']):

        # Process each player in the current team
        for player_id in team_data['participantId'].unique():

            # Extract data for the current player
            player_data = team_data[team_data['participantId'] == player_id]

            # Process each phase separately
            for phase, phase_data in player_data.groupby('Phase'):

                # Calculate the distance from this player to teammates, minute-by-minute
                avg_distances_per_minute = []

                for minute in phase_data['time'].unique():
                    # Player's distance at the current minute
                    player_distance = phase_data[phase_data['time'] == minute]['distance'].values[0]

                    # Teammates' data at the same minute
                    teammates_data = team_data[(team_data['participantId'] != player_id) & (team_data['time'] == minute)]

                    # Calculate average distance from this player to teammates
                    if not teammates_data.empty:
                        avg_teammate_distance = teammates_data['distance'].mean()
                        distance_from_teammates = abs(player_distance - avg_teammate_distance)
                        avg_distances_per_minute.append(distance_from_teammates)

                # Calculate the average split score for this phase
                avg_split_score = sum(avg_distances_per_minute) / len(
                    avg_distances_per_minute) if avg_distances_per_minute else None

                # Retrieve additional player-specific information (role, lane, rank, etc.)
                player_info = player_data.iloc[0][['individualPosition', 'lane', 'tier', 'championName', 'win']]

                # Append results to the list
                split_scores.append({
                    'matchId': match_id,
                    'participantId': player_id,
                    'phase': phase,
                    'split_score': avg_split_score,
                    'individualPosition': player_info['individualPosition'],
                    'lane': player_info['lane'],
                    'tier': player_info['tier'],
                    'championName': player_info['championName'],
                    'win': player_info['win']
                })

    # Convert the results into a DataFrame for easier readability
    split_scores_df = pd.DataFrame(split_scores)

    # Save the results to a CSV file
    outputfilename = output + 'split_scores_phases.csv'
    split_scores_df.to_csv(outputfilename, index=False)

#Split score is our way of measuring how far away a player is from their teammates
def calculate_SplitScore(df,output):
    dx = df['position_x'].diff(periods=-1)
    dy = df['position_y'].diff(periods=-1)

    df['xposition_diff'] = dx
    df['yposition_diff'] = dy

    # Calculate the distance using the Pythagorean theorem
    distances = np.sqrt(dx ** 2 + dy ** 2)
    distances.iloc[0] = 0
    df['distance'] = distances


    conditions = [df['time'] < early_phaseTr,
                  (df['time'] >= early_phaseTr) & (df['time'] < mid_phaseTr),
                  df['time'] >= mid_phaseTr]

    choices = ['1', '2', '3']  ### 1: Early Phase, 2: Mid Phase, 3: late phase
    df['Phase'] = np.select(conditions, choices, default='1')

    # Step 1: Filter data for the specified time range (15 to 30 minutes)
    df_filtered = df[(df['time'] >= 10) & (df['time'] <= 45)].copy()

    # Step 2: Define teams based on `player_id` (adjust the logic as needed)
    df_filtered['team'] = df_filtered['participantId'].apply(lambda x: 1 if x < 6 else 2)

    # Step 3: Initialize list to store each player's final average distance from team
    split_scores = []

    # Group by match and team to process each team separately
    for (match_id, team), team_data in df_filtered.groupby(['matchId', 'team']):

        # Loop over each player in the current team
        for player_id in team_data['participantId'].unique():

            # Extract data for the current player
            player_data = team_data[team_data['participantId'] == player_id]

            # Calculate the distance from this player to every other teammate, minute-by-minute
            avg_distances_per_minute = []

            # Loop over each minute in the time range
            for minute in player_data['time'].unique():
                # Player's distance at the current minute
                player_distance = player_data[player_data['time'] == minute]['distance'].values[0]

                # Teammates' data at the same minute
                teammates_data = team_data[(team_data['participantId'] != player_id) & (team_data['time'] == minute)]

                # Calculate average distance from this player to teammates at this minute
                if not teammates_data.empty:
                    avg_teammate_distance = teammates_data['distance'].mean()
                    distance_from_teammates = abs(player_distance - avg_teammate_distance)
                    avg_distances_per_minute.append(distance_from_teammates)

            # Calculate the average of the minute-by-minute distances for this player
            avg_distance_from_team = sum(avg_distances_per_minute) / len(
                avg_distances_per_minute) if avg_distances_per_minute else None

            # Retrieve additional player-specific information (role, lane, rank, etc.)
            player_info = player_data.iloc[0][['individualPosition', 'lane', 'tier', 'championName',
                                               'win']]  # Taking the first entry, assuming consistent info

            # Append results to split_scores list
            split_scores.append({
                'matchId': match_id,
                'participantId': player_id,
                'split_score': avg_distance_from_team,
                'individualPosition': player_info['individualPosition'],
                'lane': player_info['lane'],
                'tier': player_info['tier'],
                'championName': player_info['championName'],
                'win': player_info['win']
            })

    # Convert the results into a DataFrame for easier readability
    split_scores_df = pd.DataFrame(split_scores)

    outputfilename=output+'split_scores3.csv'
    split_scores_df.to_csv(outputfilename,index=False)





def plotMovementDirection_rankplayer(df, ax, title):
    if len(df) >= 2500:
        # Randomly sample 2500 rows from the DataFrame
        df = df.sample(n=2500, random_state=1)  # Set random_state for reproducibility
    else:
        # If there are fewer than 2500 rows, sample all rows
        df = df

    # Calculate the movement vectors
    dx = np.diff(df['position_x'], prepend=df['position_x'].iloc[0])
    dy = np.diff(df['position_y'], prepend=df['position_y'].iloc[0])

    # Define grid size and density threshold for dense regions
    grid_size = 10  # Smaller grid size for smaller dataset
    density_threshold = 5  # Lower threshold to highlight dense areas more easily

    # Compute 2D histogram to find dense areas
    density, x_edges, y_edges = np.histogram2d(df['position_x'], df['position_y'], bins=grid_size)

    # Filter dense areas based on density threshold
    dense_cells = density > density_threshold

    # Get grid center points for dense cells
    x_centers = (x_edges[:-1] + x_edges[1:]) / 2
    y_centers = (y_edges[:-1] + y_edges[1:]) / 2
    X, Y = np.meshgrid(x_centers, y_centers)

    # Compute average movement direction in each cell
    bin_means_x, _, _, _ = binned_statistic_2d(
        df['position_x'], df['position_y'], dx, 'mean', bins=grid_size
    )
    bin_means_y, _, _, _ = binned_statistic_2d(
        df['position_x'], df['position_y'], dy, 'mean', bins=grid_size
    )

    # Plot background scatter plot with larger point size for better visibility
    ax.scatter(df['position_x'], df['position_y'], c='blue', s=10, alpha=0.4)  # Larger size and slight opacity

    # Plot arrows in all dense regions
    for ix in range(grid_size):
        for iy in range(grid_size):
            if dense_cells[ix, iy]:
                x = X[ix, iy]
                y = Y[ix, iy]
                dx_mean = bin_means_x.T[ix, iy]
                dy_mean = bin_means_y.T[ix, iy]

                # Plot the individual arrow with adjusted width
                ax.quiver(x, y, dx_mean, dy_mean, angles='xy', scale_units='xy', scale=3, color='purple', width=0.005, headwidth=4, headlength=6)

    ax.set_xlabel("X Position")
    ax.set_ylabel("Y Position")
    ax.set_title(f"Movement Directions for Phase: {title}")
def plotMovementDirection(df, ax,title):
    # Calculate the movement vectors
    dx = np.diff(df['position_x'], prepend=df['position_x'].iloc[0])
    dy = np.diff(df['position_y'], prepend=df['position_y'].iloc[0])

    # Define grid size and density threshold for dense regions
    grid_size = 20
    density_threshold = 10  # Adjust based on data; higher values mean only denser regions get arrows

    # Compute 2D histogram to find dense areas
    density, x_edges, y_edges = np.histogram2d(df['position_x'], df['position_y'], bins=grid_size)

    # Filter dense areas based on density threshold
    dense_cells = density > density_threshold

    # Get grid center points for dense cells
    x_centers = (x_edges[:-1] + x_edges[1:]) / 2
    y_centers = (y_edges[:-1] + y_edges[1:]) / 2
    X, Y = np.meshgrid(x_centers, y_centers)

    # Compute average movement direction in each cell
    bin_means_x, _, _, _ = binned_statistic_2d(
        df['position_x'], df['position_y'], dx, 'mean', bins=grid_size
    )
    bin_means_y, _, _, _ = binned_statistic_2d(
        df['position_x'], df['position_y'], dy, 'mean', bins=grid_size
    )

    # Plot background scatter plot (optional for reference)
    ax.scatter(df['position_x'], df['position_y'], c='blue', s=1, alpha=0.2)

    # Plot arrows in all dense regions
    for ix in range(grid_size):
        for iy in range(grid_size):
            if dense_cells[ix, iy]:
                x = X[ix, iy]
                y = Y[ix, iy]
                dx_mean = bin_means_x.T[ix, iy]
                dy_mean = bin_means_y.T[ix, iy]

                # Plot the individual arrow with its respective color
                ax.quiver(x, y, dx_mean, dy_mean, angles='xy', scale_units='xy', scale=5, color='purple', headwidth=3, headlength=5)

    ax.set_xlabel("X Position")
    ax.set_ylabel("Y Position")
    ax.set_title(f"Movement Directions for Phase: {title})")


# Main plotting function
def createMovementPlots(df):
    phases = df['Phase'].unique()
    f='tier'
    wins = df[f].unique()
    wins=['CHALLENGER','EMERALD']
    ncols=2 #wins.size
    # Create a figure with 3 rows and 2 columns of subplots
    fig, axs = plt.subplots(nrows=3, ncols=ncols, figsize=(12, 15))
    axs = axs.flatten()  # Flatten the 2D array of axes for easy indexing

    for i, phase in enumerate(phases):
        for j, win in enumerate(wins):
            df_phase_win = df[(df['Phase'] == phase) & (df[f] == win)]
            title=str(win)+'_'+ str(phase)
            plotMovementDirection(df_phase_win, axs[i * 2 + j],title)

    # Adjust layout to prevent overlap
    plt.tight_layout()
    plt.show()
def plotMovementDirection_old(data,title):
    from scipy.stats import binned_statistic_2d
    """
    # Calculate the movement vectors
    dx = np.diff(data['position_x'], prepend=data['position_x'].iloc[0])
    dy = np.diff(data['position_y'], prepend=data['position_y'].iloc[0])

    # Normalize direction vectors for uniform arrow lengths (optional)
    magnitude = np.sqrt(dx ** 2 + dy ** 2)
    direction_x = dx / (magnitude + 1e-5)  # Small epsilon to prevent division by zero
    direction_y = dy / (magnitude + 1e-5)

    # Set a constant length for all arrows
    arrow_length = 1.0
    fixed_dx = direction_x * arrow_length
    fixed_dy = direction_y * arrow_length
    # Define grid size
    grid_size = 20  # Adjust based on data range and desired resolution

    # Compute the average movement direction in each cell
    bin_means_x, _, _, _ = binned_statistic_2d(
        data['position_x'], data['position_y'], direction_x, 'mean', bins=grid_size
    )
    bin_means_y, x_edges, y_edges, _ = binned_statistic_2d(
        data['position_x'], data['position_y'], direction_y, 'mean', bins=grid_size
    )

    # Plot the flow field
    plt.figure(figsize=(10, 8))
    plt.scatter(data['position_x'], data['position_y'], c='blue', s=1, alpha=0.2)

    # Get grid center points for quiver plot
    x_centers = (x_edges[:-1] + x_edges[1:]) / 2
    y_centers = (y_edges[:-1] + y_edges[1:]) / 2
    X, Y = np.meshgrid(x_centers, y_centers)

    # Plot averaged movement directions in each grid cell
    plt.quiver(X, Y, bin_means_x.T, bin_means_y.T, angles='xy', scale_units='xy', scale=6, color='red', alpha=0.8)

    # Sample points for arrows, setting sample_interval to achieve 15 labels
    max_arrows = 15
    sample_interval = max(1, len(data['position_x']) // max_arrows)

    # Plot arrows with numbers 1 to 15 for key points
    for count, i in enumerate(range(0, len(data['position_x']) - 1, sample_interval)):
        if count >= max_arrows:
            break

        x, y = data['position_x'].iloc[i], data['position_y'].iloc[i]
        dx_i, dy_i = dx[i], dy[i]

        # Plot individual arrows with a fixed length (scale of 5)
        plt.quiver(x, y, dx_i, dy_i, angles='xy', scale_units='xy', scale=6, color='purpul', headwidth=3, headlength=5)
        plt.text(x + dx_i * 0.5, y + dy_i * 0.5, str(count + 1), fontsize=12, color="black", alpha=0.9)

    plt.xlabel("X Position")
    plt.ylabel("Y Position")
    plt.title("Aggregated Movement Direction by Grid" +title)
    plt.show()
    """
# Calculate the movement vectors
    dx = np.diff(data['position_x'], prepend=data['position_x'].iloc[0])
    dy = np.diff(data['position_y'], prepend=data['position_y'].iloc[0])

    # Define grid size and density threshold for dense regions
    grid_size = 20
    density_threshold = 1000  # Adjust based on data; higher values mean only denser regions get arrows

    # Compute 2D histogram to find dense areas
    density, x_edges, y_edges = np.histogram2d(data['position_x'], data['position_y'], bins=grid_size)

    # Filter dense areas based on density threshold
    dense_cells = density > density_threshold

    # Get grid center points for dense cells
    x_centers = (x_edges[:-1] + x_edges[1:]) / 2
    y_centers = (y_edges[:-1] + y_edges[1:]) / 2
    X, Y = np.meshgrid(x_centers, y_centers)

    # Compute average movement direction in each cell
    bin_means_x, _, _, _ = binned_statistic_2d(
        data['position_x'], data['position_y'], dx, 'mean', bins=grid_size
    )
    bin_means_y, _, _, _ = binned_statistic_2d(
        data['position_x'], data['position_y'], dy, 'mean', bins=grid_size
    )

    # Plot background scatter plot (optional for reference)
    plt.figure(figsize=(10, 8))
    plt.scatter(data['position_x'], data['position_y'], c='blue', s=1, alpha=0.2)

    # Initialize variables for resultant vectors
    resultant_dx = 0
    resultant_dy = 0

    # Store positions of dense cells for resultant vector calculation
    x_positions = []
    y_positions = []

    # Plot arrows in all dense regions and calculate resultant vector
    for ix in range(grid_size):
        for iy in range(grid_size):
            if dense_cells[ix, iy]:
                x = X[ix, iy]
                y = Y[ix, iy]
                dx_mean = bin_means_x.T[ix, iy]
                dy_mean = bin_means_y.T[ix, iy]

                # Plot the individual arrow with its respective color
                plt.quiver(x, y, dx_mean, dy_mean, angles='xy', scale_units='xy', scale=5, color='purple', headwidth=3,
                           headlength=5)

    plt.xlabel("X Position")
    plt.ylabel("Y Position")
    plt.title("Movement Direction in Dense Regions with Ordered Path" +title)
    plt.show()



def plotMovementSequenceWithNumbers(df, ax, title):
        # Ensure the DataFrame is sorted by time (if not already)
        df = df.sort_values(by='time')  # Replace 'time' with your actual time column name if different
        density_threshold = 1000
        # Get positions
        # Get positions
        x_positions = df['position_x'].values
        y_positions = df['position_y'].values

        # Compute the 2D histogram to find dense areas
        grid_size = 20  # Define grid size
        density, x_edges, y_edges = np.histogram2d(x_positions, y_positions, bins=grid_size)

        # Filter dense areas based on density threshold
        dense_cells = density > density_threshold

        # Get grid center points for dense cells
        x_centers = (x_edges[:-1] + x_edges[1:]) / 2
        y_centers = (y_edges[:-1] + y_edges[1:]) / 2
        X, Y = np.meshgrid(x_centers, y_centers)

        # Plot background scatter plot for all points
        ax.scatter(x_positions, y_positions, c='blue', s=1, alpha=0.3)  # Adjust size and opacity

        # Store the coordinates of dense cells where numbers will be placed
        dense_points_x = []
        dense_points_y = []

        # Plot numbers in all dense regions
        counter = 1  # Initialize counter for numbering
        for ix in range(grid_size):
            for iy in range(grid_size):
                if dense_cells[ix, iy]:
                    # Collect coordinates for line plot
                    dense_points_x.append(X[ix, iy])
                    dense_points_y.append(Y[ix, iy])

                    # Plot the order number at the center of the dense cell
                    #ax.text(X[ix, iy], Y[ix, iy], str(counter), fontsize=10, ha='center', va='center',
                    #        color='red', fontweight='bold')  # Bold red text
                    counter += 1  # Increment the counter for the next number

            # Use spline interpolation to create a smooth line
        if len(dense_points_x) > 2:  # Ensure there are enough points for a spline
            tck, u = splprep([dense_points_x, dense_points_y], s=0)  # Fit spline to the dense points
            smooth_points = splev(np.linspace(0, 1, 100), tck)  # Generate 100 interpolated points along the spline
            ax.plot(smooth_points[0], smooth_points[1], color='purple', linestyle='-', linewidth=1)

        ax.set_xlabel("X Position")
        ax.set_ylabel("Y Position")
        ax.set_title(title)
def plotTimeAveragedMovement(df,ax, title,phase):
    # Group by each minute, calculate mean and std for x and y positions
    grouped = df.groupby('time').agg(
        mean_x=('position_x', 'mean'),
        mean_y=('position_y', 'mean'),
        std_x=('position_x', 'std'),
        std_y=('position_y', 'std')
    ).reset_index()

    background_image = plt.imread('../RealLOLmap.png')

    # Prepare data for smooth line plotting
    mean_x = grouped['mean_x'].values
    mean_y = grouped['mean_y'].values
    std_x = grouped['std_x'].values
    std_y = grouped['std_y'].values

    # Interpolated smooth path
    time_points = np.linspace(0, len(mean_x) - 1, 200)
    spline_x = make_interp_spline(np.arange(len(mean_x)), mean_x, k=3)(time_points)
    spline_y = make_interp_spline(np.arange(len(mean_y)), mean_y, k=3)(time_points)

    if phase==1:
        min_x=0
        min_y=0
        max_x=8000
        max_y=8000
    elif phase==2:
        min_x=5000
        min_y=5000
        max_x=8000
        max_y=8000
    else:
        min_x=0
        min_y=0
        max_x=15000
        max_y=15000
    # Calculate the bounding box of the points to zoom in

    """min_x = min(mean_x)
    max_x = max(mean_x)
    min_y = min(mean_y)
    max_y = max(mean_y)

    # Apply a small padding around the bounding box for better view
    padding = 500  # adjust this as needed
    min_x -= padding
    max_x += padding
    min_y -= padding
    max_y += padding"""


    # Display the background image with zoomed-in region
    ax.imshow(background_image, extent=[min_x, max_x, min_y, max_y], alpha=0.7, aspect='auto')

    # Plot the smoothed line
    ax.plot(spline_x, spline_y, color='purple', linestyle='-', linewidth=1, label='Smoothed Path')



    # Add arrows for smooth path to indicate movement direction
    arrow_interval = 20
    for i in range(0, len(spline_x) - arrow_interval, arrow_interval):
        dx = spline_x[i + arrow_interval] - spline_x[i]
        dy = spline_y[i + arrow_interval] - spline_y[i]
        ax.arrow(spline_x[i], spline_y[i], dx, dy, head_width=5, head_length=5, fc='purple', ec='purple', lw=2, width=50)

    # Highlight start and end points
    ax.plot(mean_x[0], mean_y[0], color='green', marker='o', markersize=8, label='Start')
    ax.plot(mean_x[-1], mean_y[-1], color='blue', marker='o', markersize=8, label='End')

    # Adjust axes limits to match the zoomed-in bounding box
    ax.set_xlim(min_x, max_x)
    ax.set_ylim(min_y, max_y)

    ax.set_title(title)



def plotTimeAveragedMovement2(df, ax,title):
    x_offset = 800
    y_offset = 300
    # Group the data by each minute and calculate the mean and standard deviation for x and y positions
    grouped = df.groupby('time').agg(
        mean_x=('position_x', 'mean'),
        mean_y=('position_y', 'mean'),
        std_x=('position_x', 'std'),
        std_y=('position_y', 'std')
    ).reset_index()
    img = mpimg.imread('../RealLOLmap.jpg')
    # Get the mean positions and standard deviations
    mean_x = grouped['mean_x'].values + x_offset
    mean_y = grouped['mean_y'].values + y_offset
    std_x = grouped['std_x'].values
    std_y = grouped['std_y'].values

    # Generate an interpolation for smoother lines
    # Create a range of "time" points for interpolation
    time_points = np.linspace(0, len(mean_x) - 1, 200)  # 300 interpolated points
    spline_x = make_interp_spline(np.arange(len(mean_x)), mean_x, k=3)(time_points)  # Smooth x-coordinates
    spline_y = make_interp_spline(np.arange(len(mean_y)), mean_y, k=3)(time_points)  # Smooth y-coordinates

    ax.imshow(img, extent=[0, 15000, 0, 15000], alpha=0.7, aspect='auto')
    # Plot the smooth line
    ax.plot(spline_x, spline_y, color='purple', linestyle='-', linewidth=3, label='Smoothed Path')

    # Plot circles with radius of 2 * std deviation around each mean point
   # for i in range(len(mean_x)):
        # Circle radius as 2 * std for each point (max of std_x and std_y to make circle radius uniform)
       # circle_radius = 2 * max(std_x[i], std_y[i])
      #  circle = plt.Circle((mean_x[i], mean_y[i]), circle_radius, color='purple', fill=False, linestyle='--', alpha=0.5)
       # ax.add_patch(circle)

    arrow_interval = 2 # Set interval for arrow placement
    for i in range(0, len(spline_x) - arrow_interval, arrow_interval):  # Stop before reaching end of arrays
        # Find the direction of the movement at each point by calculating the derivative
        dx = spline_x[i + arrow_interval] - spline_x[i]
        dy = spline_y[i + arrow_interval] - spline_y[i]

        # Normalize the vector
        magnitude = np.sqrt(dx ** 2 + dy ** 2)
        dx /= magnitude
        dy /= magnitude

        # Plot the arrow at the point
      #  ax.annotate('', xy=(spline_x[i + arrow_interval], spline_y[i + arrow_interval]),
      #              xytext=(spline_x[i], spline_y[i]),
      #              arrowprops=dict(arrowstyle='->', color='red', lw=1.5))

       # Add an arrow on the smooth line at the calculated position
        ax.arrow(spline_x[i], spline_y[i], dx, dy, head_width=12, head_length=10, fc='purple', ec='purple', lw=2,width=1.5)

    # Highlight the starting and ending points
    ax.plot(mean_x[0], mean_y[0], color='green', marker='o', markersize=8, label='Start', zorder=5)  # Start point
    ax.plot(mean_x[-1], mean_y[-1], color='blue', marker='o', markersize=8, label='End', zorder=5)  # End point

    ax.set_xlim(4000, 12000)
    ax.set_ylim(4000, 12000)
    ax.set_xlabel("X Position")
    ax.set_ylabel("Y Position")
    ax.set_title(title)
    #ax.legend()



def LICpatternMovement(file):

    df = pd.read_csv(file)
    df = df[df['participantId'] < 6]
    df=df[df['individualPosition']=='JUNGLE']
    phases = df['Phase'].unique()
    f='win'
    filters = df[f].unique()
    #filters=['CHALLENGER','EMERALD','SILVER']
    ncols=2#filters.size()
    # Create a figure with 3 rows and 2 columns of subplots
    fig, axs = plt.subplots(nrows=3, ncols=ncols, figsize=(15, 10))
    axs = axs.flatten()  # Flatten the 2D array of axes for easy indexing

    fig.patch.set_alpha(0)
    for ax in axs:
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
    #fig, ax = plt.subplots(figsize=(8, 6))


    #createMovementPlots(df)
    for i, phase in enumerate(phases):
        for j, filter in enumerate(filters):
            #df_phase_win = df[(df[f] == filter)]
            df_phase_win = df[(df['Phase'] == phase) & (df[f] == filter)]

            title = str(filter) + '_' + str(phase)
            #plotMovementDirection(df_phase_win, axs[i * 2 + j], title)

            #plotMovementDirection_rankplayer(df_phase_win, axs[i * 2 + j], title=title)


           # plotMovementDirection_old(df_phase_win,title)

            lic_result=flow_like_texture_movement(df_phase_win)
            alpha_shape=compute_alphaShape(df_phase_win,title,axs[i * 2 + j],filter)
            combine_alphasShapandLLC(lic_result, alpha_shape,title,axs[i * 2 + j])
            # Example usage with your DataFrame

            #plotMovementSequenceWithNumbers(df_phase_win, axs[i * 2 + j],title)
            #plotTimeAveragedMovement(df_phase_win, axs[i * 2 + j],title)
            #plotTimeAveragedMovement(df_phase_win, axs[i * 3 + j], title,phase)
    # Gather legend items from one of the subplots (all subplots share the same legend items)
    #handles, labels = axs[0, 0].get_legend_handles_labels()
    #fig.legend(handles, labels, loc="center right", borderaxespad=1, title="Legend")

    # Adjust layout to fit legend on the right
    #plt.tight_layout(rect=[0, 0, 0.85, 1])
    plt.show()
           # fig, axs = plt.subplots(nrows=1, ncols=ncols, figsize=(12, 8))

def flow_like_texture_movement(data):
    # Sample data (replace this with your actual data)
    x, y = data['position_x'], data['position_y']
    #dx = np.diff(np.concatenate(([x[0]], x)))
    dx = x.diff().fillna(0)
    #dx = np.diff(x, prepend=x[0])
    dy = y.diff().fillna(0)

    # Grid setup
    grid_x, grid_y = np.mgrid[x.min():x.max():500j, y.min():y.max():500j]
    points = np.array(list(zip(x, y)))
    vectors = np.array(list(zip(dx, dy)))

    # Interpolate onto the grid
    grid_vectors_x = griddata(points, vectors[:, 0], (grid_x, grid_y), method='cubic', fill_value=0)
    grid_vectors_y = griddata(points, vectors[:, 1], (grid_x, grid_y), method='cubic', fill_value=0)

    # Normalize vectors
    mag = np.sqrt(grid_vectors_x ** 2 + grid_vectors_y ** 2)
    u = grid_vectors_x / (mag + 1e-5)
    v = grid_vectors_y / (mag + 1e-5)

    # Generate noise texture for LIC
    noise = np.random.rand(*u.shape)

    # Apply a simple LIC by averaging along streamlines
    # Adjust `kernel_size` and `blur` for different visual effects
    kernel_size = 10
    for i in range(1, kernel_size):
        noise = noise + np.roll(noise, i, axis=0) * u + np.roll(noise, i, axis=1) * v
    lic_result = gaussian_filter(noise, sigma=1)

    # Load the background image
    background_image = plt.imread('../RealLOLmap.png')  # Update the path to your image file

    # Create a figure and axis
    fig, ax = plt.subplots(figsize=(10, 8))

    # Display the background image
    ax.imshow(background_image, extent=(x.min(), x.max(), y.min(), y.max()),
              aspect='auto')  # Adjust extent to fit your data

    # Plot LIC result on top of the background
    plt.imshow(lic_result, cmap='gray', extent=(x.min(), x.max(), y.min(), y.max()),
               alpha=0.5)  # Adjust alpha for transparency

    # Set titles and labels
    ax.set_title("Player Movement Flow with Background")
    ax.set_xlabel("X Position")
    ax.set_ylabel("Y Position")

    # Show the plot
    plt.show()

    return (lic_result)

def compute_alphaShape(data,title,ax,f):
    # Load the background image
    background_image = plt.imread('../RealLOLmap.png')  # Update this path to your image file

    def plot_with_background(ax, title):
        # Display the background image
        ax.imshow(background_image, extent=(0, 15000, 0, 15000), aspect='auto')  # Adjust extent to fit your data
        ax.set_title(title)
        ax.set_xlabel("X Position")
        ax.set_ylabel("Y Position")

    offsetx=0
    offsety = 0
    data['position_x'] = data['position_x'] + offsetx
    data['position_y'] = data['position_y'] + offsety

    # Prepare data points
    points = [(x, y) for x, y in zip(data['position_x'], data['position_y'])]

    # Compute the alpha shape
    alpha = 0.1  # adjust this parameter as needed
    alpha_shape = alphashape.alphashape(points, alpha)

    # Plot the alpha shape
    #fig, ax = plt.subplots(figsize=(10, 8))

    # Call the background plotting function
    #plot_with_background(ax, title)

    # Scatter plot for player movements
    """if f==1:
        ax.scatter(data['position_x'], data['position_y'], s=20, c='blue', alpha=0.3, label="Player Movements")
    else:
        ax.scatter(data['position_x'], data['position_y'], s=20, c='red', alpha=0.3, label="Player Movements")
"""
    if f == 1:
        c='blue'
    else:
        c='red'

    # Plotting the alpha shape
    if isinstance(alpha_shape, Polygon):
        x, y = alpha_shape.exterior.xy
        ax.plot(x, y, color=c, linewidth=2, label="Alpha Shape Boundary")
    else:
        for geom in alpha_shape.geoms:  # For multiple polygons
            x, y = geom.exterior.xy
            ax.plot(x, y, color=c, linewidth=2)

    # Finalizing the plot
   # ax.legend()
    #plt.show()

    return alpha_shape

def combine_alphasShapandLLC(lic_result,alpha_shape,title,ax):
    # Load the background image
    #background_image = plt.imread('../RealLOLmap.png')  # Update this path to your image file

    def plot_with_background(ax, title):
        # Display the background image
        #ax.imshow(background_image, extent=(0, 15000, 0, 15000), aspect='auto')  # Adjust extent to fit your data
        ax.set_title(title)
        ax.set_xlabel("X Position")
        ax.set_ylabel("Y Position")

    # Assuming 'lic_result' is already computed and 'alpha_shape' is obtained from previous code
    # Create a figure and axis
    #fig, ax = plt.subplots(figsize=(12, 10))

    # Call the background plotting function
    #plot_with_background(ax, title)



    # Plotting the alpha shape
    if isinstance(alpha_shape, Polygon):
        x, y = alpha_shape.exterior.xy
        ax.plot(x, y, color='green', linewidth=2, label="Alpha Shape Boundary")
    else:
        for geom in alpha_shape.geoms:
            x, y = geom.exterior.xy
            ax.plot(x, y, color='green', linewidth=2)

    # Finalizing the plot
    ax.legend()

    #plt.show()

def Visualize_LolMapBasedonPosition(df,role,phase,k):
    background_image = plt.imread('LOLmap.jpg')
    x = df['position_x']
    y = df['position_y']

    extent = np.min(x), np.max(x), np.min(y), np.max(y)


    for i in range(k):
        df_cluster=df[df['cluster']==i]
        # Add the background image
        plt.imshow(background_image, extent=extent, aspect='auto')
        plt.scatter(df_cluster['position_x'], df_cluster['position_y'],c=color_list[i])


        plt.xlabel('X-axis')
        plt.ylabel('Y-axis')
        plt.title(f'Scatter Plot for {role} - {phase}-CLUSTER -{i}')

        # Display the plot
        plt.show()



def elbow(X, kmeans_kwargs):
    cs = []
    num_cluster = int(len(X) - 2)
    max_cluster = 6  # no need to have more than 6 cluster
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


def ClusteringData(X,df):
    kmeans_kwargs = {
        "init": "k-means++",
        "n_init": 2,
        "max_iter": 50000,
        "random_state": 42,
    }



    # Call elbow algoritm to find number of cluster
    num_cluster_el = elbow(X, kmeans_kwargs)
    #num_cluster_si = silhouette(X)
    #num_cluster_si=3
    #num_cluster_el=3
    num_cluster_si=None
    print('num_cluster for elbow ', num_cluster_el)
    print('num_cluster for silhouette ', num_cluster_si)

        # Call Clustring algoritm to clustr the data and label the data
    if num_cluster_si is not None:

        df_result,k=make_clustring(df, X, kmeans_kwargs, num_cluster_si, output_file)
    else:
        if num_cluster_el is not None:
            df_result,k=make_clustring(df, X, kmeans_kwargs, num_cluster_el, output_file)

    return (df_result,k)


def silhouette (X):
    # Create a list to store Silhouette scores
    silhouette_scores = []

    # Try different values of k
    for k in range(2, 11):
        kmeans = KMeans(n_clusters=k, random_state=42)
        labels = kmeans.fit_predict(X)
        silhouette_scores.append(silhouette_score(X, labels))

    # Plot the Silhouette scores for different values of k
    plt.plot(range(2, 11), silhouette_scores)
    plt.xlabel('Number of Clusters')
    plt.ylabel('Silhouette Score')
    plt.show()

    # *******************************************************


def make_clustring(df_original, df, kmeans_kwargs, k, output_file):
    #k = 3

    kmeans = KMeans(n_clusters=k, **kmeans_kwargs)
    # X=df['LapTime'].values.reshape(-1,1)
    X = df

    kmeans.fit(X)

    labels = kmeans.fit_predict(X)

    # Adding the results to a new column in the dataframe
    df_original["cluster"] = labels  # kmeans.labels_

    # making readable labels for cluster
    # df_original=(Create_clusterLabel(df_original))

    df_original.to_csv(output_file, index=False)
    return(df_original,k)


def assign_t2(df):
    current_value = None
    count = 0

    for i in range(len(df)):
        if df.loc[i, 'participantId'] == current_value:
            count += 1
        else:
            current_value = df.loc[i, 'participantId']
            count = 1

        if count == 1:
            df.loc[i, 'Phase'] = '1'
        elif count == 2:
            df.loc[i, 'Phase'] = '2'
        elif count == 3:
            df.loc[i, 'Phase'] = '3'

    return df


def calculate_TotalMovementandPlot(file1,path):
    df=pd.read_csv(file1)
    outputfilename=path+'TotalMovementAndPlot.csv'
    df = df[df['participantId'] < 6]
    # Step 1: Calculate the movement (distance between consecutive positions)
    df['delta_x'] = df['position_x'].diff()
    df['delta_y'] = df['position_y'].diff()
    df['movement'] = np.sqrt(df['delta_x'] ** 2 + df['delta_y'] ** 2)

    outputfilename2=path+'MovementPerParticipant.csv'
    df.to_csv(outputfilename2, index=False)

    for phase in df['Phase'].unique():
        filtered_df = df[df['Phase']==phase]
        # Group by 'time', 'tier', and 'phase' and calculate the mean of 'm'
        grouped_df = filtered_df.groupby(['tier','time']).agg(avg_m=('movement', 'mean')).reset_index()

        # Calculate the cumulative sum of avg_m for each phase and tier
        #grouped_df['cumulative_avg_m'] = grouped_df.groupby('tier')['avg_m'].cumsum()

        # Save the data to a CSV file
        if phase==3 :
          grouped_df.to_csv(outputfilename, index=False)

        # Step 4: Plot the results
        plt.figure(figsize=(10, 6))
        plt.figure(figsize=(10, 6))
        for tier in grouped_df['tier'].unique():
            tier_df = grouped_df[grouped_df['tier'] == tier]
            plt.plot(tier_df['time'], tier_df['avg_m'], marker='o', linestyle='-',label=f'Tier: {tier}')

        plt.xlabel('Time')
        plt.ylabel('Total Movement')
        plt.title(f'Total Movement during {phase}')
        plt.legend()
        plt.show()


def distance_statAnalysis(filename1,filename2,filename3):
    df = pd.read_csv(filename1)
    #df['role'] = df['role'].replace('NONE', 'JUNGLE')
    #df['lane'] = df['lane'].replace('NONE', 'JUNGLE')


   # getNumber_ofzoneChange(filename3)
    #df=assign_t2(df)
  #

    total_distanceBetweenPlayersperTeam(df)

   # total_movement_perPhase(df)

    total_movement_allPhaseRole(df,'allRanks')
    total_movement_allPhaseRole_WinSeperation(df,'allRanks')

   # total_movement_allPhaseRole_BoxPlot(df)
   # total_movement_allPhaselane(df,'all')
    ############
    df2 = pd.read_csv(filename2)
    total_movement_allrankPlayer(df2)
    total_movement_allPhaserankPlayer(df2,'all')

    """for cls in df['tier'].unique():
        df_tier = df[df['tier'] == cls]
        total_movement_allPhaseRole(df_tier,cls)"""
    # Apply the function to the DataFrame
    #df = assign_t2(df)

    #df.to_csv(filename, index=False)

def find_numberofZoneChanged(filename1):
    df=pd.read_csv(filename1)
    new_df=assign_PhasetoPositionrowData(df)
    new_df['Totalnumberof_Zonechanged'] = new_df.groupby(['match_id', 'participantId'])['assigned_zone'].transform(lambda x: x.diff().ne(0).sum())
    new_df['Phasednumberof_Zonechanged'] = new_df.groupby(['match_id', 'participantId','Phase'])['assigned_zone'].transform(lambda x: x.diff().ne(0).sum())
    new_df.drop(['numberof_Zonechanged','Unnamed: 0.1','Unnamed: 0'], axis=1, inplace=True)

    new_df.to_csv(filename1,index=True)
   # df_original = df.copy()
   # changes = df.groupby(['matchId', 'participantId'])['cluster'].apply(lambda x: x.diff().ne(0).cumsum())

def total_distanceBetweenPlayersperTeam(df):


    #####total distance between players in each team

    df['total_team movement'] = df.apply(lambda row: row['total_movement_team1'] if row['participantId'] < 6 else row['total_movement_team2'], axis=1)

    df_grouped=df.groupby(['matchId','win']).agg({'total_team movement': 'last'}).reset_index()
    #####total movement of each team
    avg_t = df_grouped.groupby('win')['total_team movement'].mean()
    output_filename=outputdata_path+'total distance between players in each team.csv'
    output_filename2 = outputdata_path + 'total distance between players in each teamRowdata.csv'
    df.to_csv(output_filename2)
    avg_t.to_csv(output_filename)
    # Create a bar graph
    avg_t.plot(kind='bar', color=['blue', 'red'])
    plt.xlabel('win')
    plt.ylabel('Average total distance between players')
    plt.title('Average total distance between players in each team')
    plt.xticks(ticks=[0, 1], labels=['loss', 'win'], rotation=0)
    plt.show()


    #################################
    #####total movement in each team
    df_grouped=df.groupby(['matchId','win']).agg({'totalmovement distance': 'sum'}).reset_index()
    #####total movement of each team
    avg_t = df_grouped.groupby('win')['totalmovement distance'].mean()
    output_filename=outputdata_path+'Average total_team movement for winner and looser.csv'

    avg_t.to_csv(output_filename)
    # Create a bar graph
    avg_t.plot(kind='bar', color=['blue', 'red'])
    plt.xlabel('win')
    plt.ylabel('Average total_team movement')
    plt.title('Average total_team movement for winner and looser')
    plt.xticks(ticks=[0, 1], labels=['loss', 'win'], rotation=0)
    plt.show()


    #################################
def total_movement_perPhase(df):

    for cls in df['Phase'].unique():
        df_phase = df[df['Phase'] == cls]
        mean_t_per_r = df_phase.groupby('individualPosition')['totalmovement distance per phase'].mean()
        print(f'Mean value of total movement distance in {cls} for each role')
        print(mean_t_per_r)

        # Create a bar graph
        mean_t_per_r.plot(kind='bar', color='blue')
        plt.xlabel('R')
        plt.ylabel('Mean total movement distance')
        plt.title(f'Mean total movement distance per role for {cls}')
        plt.xticks(rotation=0)
        plt.show()

def total_movement_allPhaseRole_BoxPlot(df):
    # Create a figure and axes
    fig, ax = plt.subplots(figsize=(14, 8))  # Increase figure size

    # Create a boxplot
    sns.boxplot(x='individualPosition', y='totalmovement distance per phase', hue='Phase', data=df, ax=ax, palette={1: 'red', 2: 'blue', 3: 'green'})

    # Set labels and title
    ax.set_xlabel('individualPosition')
    ax.set_ylabel('Value of C')
    ax.set_title('Boxplot of Column C for Each Role and Phase')

    # Display the legend
    ax.legend(title='Phase')

    # Show the plot
    plt.show()
def total_movement_allPhaserankPlayer(df,rankname):
    plt.figure(figsize=(14, 8))
    ranks_order = ["IRON", "BRONZE", "SILVER", "GOLD", "PLATINUM", "EMERALD","DIAMOND", "MASTER", "GRANDMASTER", "CHALLENGER"]

    custom_colors = ['r', 'b', 'g']
    sns.barplot(x='tier', y='totalmovement distance per phase', hue='Phase', data=df, order=ranks_order,palette=custom_colors)

    grouped = df.groupby(['tier', 'Phase'])['totalmovement distance per phase']


    mean_values = grouped.mean().reset_index(name='mean_value')
    std_values = grouped.sem().reset_index(name='sem')

    # Merge mean and standard deviation data
    result = pd.merge(mean_values, std_values, on=['tier', 'Phase'])

    # Sort the result DataFrame based on the order of ranks
    result['tier'] = pd.Categorical(result['tier'], categories=ranks_order, ordered=True)
    result.sort_values(by='tier', inplace=True)

    # Save the sorted mean values to a CSV file
    filename=outputdata_path+'total_movement_allPhaserankPlayer.csv'
    result.to_csv(filename, index=False)

    plt.xlabel('Rank')
    plt.ylabel('average of player total movements')
    plt.title('Bar Plot of average of player total movements for different Ranks and Three Phases')
    plt.legend(title='Phase')
    plt.xticks(rotation=45)


    plt.show()

def total_movement_allrankPlayer(df):
    plt.figure(figsize=(14, 8))
    ranks_order = ["IRON", "BRONZE", "SILVER", "GOLD", "PLATINUM", "EMERALD","DIAMOND", "MASTER", "GRANDMASTER", "CHALLENGER"]

    plt.figure(figsize=(12, 6))

    df_grouped=df.groupby(['matchId','tier','Phase']).agg({'totalmovement distance': 'last'}).reset_index()
    #####total movement of each team
    avg_t = df_grouped.groupby('tier')['totalmovement distance'].mean()
    output_filename=outputdata_path+'average total movement in different ranks.csv'

    # Save the sorted mean values to a CSV file

    avg_t.to_csv(output_filename, index=True)

    # Create the bar plot
    fig, ax = plt.subplots()

    # Plot the bar graph
    ax.bar(avg_t['tier'], avg_t['totalmovement distance'])

    # Customizing the plot
    ax.set_xlabel('Rank')
    ax.set_ylabel('average total movement')
    ax.set_title('average total movement in different ranks')

    # Show the plot
    plt.show()

    custom_colors = ['r', 'b', 'g']
    sns.barplot(x='tier', y='totalmovement distance per phase', hue='Phase', data=df, order=ranks_order,palette=custom_colors)

    grouped = df.groupby(['tier', 'Phase'])['totalmovement distance per phase']


    mean_values = grouped.mean().reset_index(name='mean_value')
    std_values = grouped.sem().reset_index(name='sem')

    # Merge mean and standard deviation data
    result = pd.merge(mean_values, std_values, on=['tier', 'Phase'])

    # Sort the result DataFrame based on the order of ranks
    result['tier'] = pd.Categorical(result['tier'], categories=ranks_order, ordered=True)
    result.sort_values(by='tier', inplace=True)

    # Save the sorted mean values to a CSV file
    filename=outputdata_path+'total_movement_allPhaserankPlayer.csv'
    result.to_csv(filename, index=False)

    plt.xlabel('Rank')
    plt.ylabel('average of player total movements')
    plt.title('Bar Plot of average of player total movements for different Ranks and Three Phases')
    plt.legend(title='Phase')
    plt.xticks(rotation=45)


    plt.show()

def getNumber_ofzoneChange(filename):
    df=pd.read_csv(filename)
   # df['individualPosition'] = df['individualPosition'].replace('NONE', 'JUNGLE')
   # df['lane'] = df['lane'].replace('NONE', 'JUNGLE')
   # df.to_csv(filename)
    ####TOTAL ZONE CHANGES FOR winner/lost ###################################

    plt.figure(figsize=(12, 6))


    # Group by column 'A' and sum column 'B'
    grouped = df.groupby('win')['Totalnumberof_Zonechanged'].mean().reset_index()  #

     # Save the sorted mean values to a CSV file
    filename = outputdata_path + 'Total Zone changes-winloss.csv'
    grouped.to_csv(filename, index=False)

    grouped.plot(kind='bar', color=['blue', 'red'])
    # Plot the bar graph
    plt.bar(grouped['win'], grouped['Totalnumberof_Zonechanged'])

    # Customizing the plot
    plt.xlabel('win')
    plt.ylabel('Average of Number of times zone changes')
    plt.title('Average of Number of times zone changes for winner and looser teams')

    # Show the plot
    plt.show()

    ####TOTAL ZONE CHANGES FOR DIFFERENT ROLES ###################################

    plt.figure(figsize=(12, 6))

    grouped = df.groupby('individualPosition')['Totalnumberof_Zonechanged'].mean().reset_index()  #

    # Save the sorted mean values to a CSV file
    filename = outputdata_path + 'Total Zone changes-roles.csv'
    grouped.to_csv(filename, index=False)

    # Create the bar plot
    fig, ax = plt.subplots()

    # Plot the bar graph
    ax.bar(grouped['individualPosition'], grouped['Totalnumberof_Zonechanged'])

    # Customizing the plot
    ax.set_xlabel('individualPosition')
    ax.set_ylabel('Average of Number of times zone changes')
    ax.set_title('Average of Number of times zone changes for different roles')

    # Show the plot
    plt.show()

    #TOTAL NUMBER OF ZONE CHANGES IN EACH ROLE PER PHASE#################################
    plt.figure(figsize=(14, 8))


    sns.barplot(x='individualPosition', y='Phasednumberof_Zonechanged', hue='Phase', data=df)

    grouped = df.groupby(['individualPosition', 'Phase'])['Phasednumberof_Zonechanged']

    mean_values = grouped.mean().reset_index(name='mean_value')
    std_values = grouped.sem().reset_index(name='std_dev')

    # Merge mean and standard deviation data
    result = pd.merge(mean_values, std_values, on=['individualPosition', 'Phase'])

    # Sort the result DataFrame based on the order of ranks
    result['individualPosition'] = pd.Categorical(result['individualPosition'],  ordered=True)

    result.sort_values(by='individualPosition', inplace=True)

    # Save the sorted mean values to a CSV file
    filename = outputdata_path + 'numberofZonechangedPerPhasePerRole.csv'
    result.to_csv(filename, index=False)

    plt.xlabel('individualPosition')
    plt.ylabel('Average of Number of times zone changes')
    plt.title('Average of Number of times zone changes per Phase/Role')
    plt.legend(title='Phase')
    plt.xticks(rotation=45)

    plt.show()
    ###################################

    plt.figure(figsize=(16, 6))

    ranks_order = ["IRON", "BRONZE", "SILVER", "GOLD", "PLATINUM", "EMERALD", "DIAMOND", "MASTER", "GRANDMASTER",
                   "CHALLENGER"]

    # Group by column 'A' and sum column 'B'
    grouped = df.groupby('tier')['Totalnumberof_Zonechanged'].mean().reset_index()#

    # Convert 'A' to a categorical type with the specified order
    grouped['tier'] = pd.Categorical(grouped['tier'], categories=ranks_order, ordered=True)

    # Save the sorted mean values to a CSV file
    filename = outputdata_path + 'Total Average of Zonechanges PerPlayerRank.csv'
    grouped.to_csv(filename, index=False)


    # Plot the bar graph
    plt.plot(kind='bar')
    plt.bar(grouped['tier'], grouped['Totalnumberof_Zonechanged'])

    # Customizing the plot
    plt.xlabel('Rank')
    plt.ylabel('Average of Number of times zone changes')
    plt.title('Average of Number of times zone changes per Rank')

    # Show the plot
    plt.show()
    ###################################################################
    plt.figure(figsize=(14, 8))

    ranks_order = ["IRON", "BRONZE", "SILVER", "GOLD", "PLATINUM", "EMERALD","DIAMOND", "MASTER", "GRANDMASTER", "CHALLENGER"]

    sns.barplot(x='tier', y='Phasednumberof_Zonechanged', hue='Phase', data=df, order=ranks_order)

    grouped = df.groupby(['tier', 'Phase'])['Phasednumberof_Zonechanged']


    mean_values = grouped.mean().reset_index(name='mean_value')
    std_values = grouped.sem().reset_index(name='std_dev')

    # Merge mean and standard deviation data
    result = pd.merge(mean_values, std_values, on=['tier', 'Phase'])

    # Sort the result DataFrame based on the order of ranks
    result['tier'] = pd.Categorical(result['tier'], categories=ranks_order, ordered=True)

    result.sort_values(by='tier', inplace=True)

    # Save the sorted mean values to a CSV file
    filename=outputdata_path+'numberofZonechangedPerPhasePerPlayerRank.csv'
    result.to_csv(filename, index=False)

    plt.xlabel('Rank')
    plt.ylabel('Average of Number of times zone changes')
    plt.title('Bar Plot of Average of Number of zone changes for different Ranks and Three Phases')
    plt.legend(title='Phase')
    plt.xticks(rotation=45)


    plt.show()
###############################################################
def total_movement_allPhaselane(df,rankname):
    aggregated_data = df.groupby(['lane', 'Phase'])['totalmovement distance per phase'].mean().unstack(fill_value=0)

    filename=outputdata_path+'total_movement_allPhaselane' + '_'+ rankname+'.csv'
    aggregated_data.to_csv(filename)
    # Plot the bar graph
    fig, ax = plt.subplots(figsize=(10, 6))

    # Define positions and width
    positions = np.arange(len(aggregated_data))
    bar_width = 0.2

    # Plot each phase with different colors
    ax.bar(positions - bar_width, aggregated_data[1], width=bar_width, color='red', label='EarlyGame')
    ax.bar(positions, aggregated_data[2], width=bar_width, color='blue', label='MidGame')
    ax.bar(positions + bar_width, aggregated_data[3], width=bar_width, color='green', label='LateGame')

    # Set labels and title
    ax.set_xlabel('Lane')
    ax.set_ylabel('Average of total movement distance ')
    ax.set_title(f'Average of total movement distance for Each Role and Phase for {rankname} Players')
    ax.set_xticks(positions)
    ax.set_xticklabels(aggregated_data.index)
    ax.legend()

    # Show the plot
    plt.show()

def total_movement_allPhaseRole_WinSeperation(df,rankname):
    # Step 1: Group by 'phase', 'individualPosition', and 'w' to calculate the average of 'm'
    grouped = df.groupby(['Phase', 'individualPosition', 'win'])['totalmovement distance per phase'].mean().reset_index()

    # Step 2: Pivot the data to have 'w' as columns
    pivot = grouped.pivot_table(index=['Phase', 'individualPosition'], columns='win', values='totalmovement distance per phase', fill_value=0).reset_index()

    # Step 3: Calculate total average and percentages for w=0 and w=1
    pivot['total_avg_m'] = pivot[0] + pivot[1]
    pivot['percentage_w0'] = pivot[0] / pivot['total_avg_m']
    pivot['percentage_w1'] = pivot[1] / pivot['total_avg_m']

    filename = outputdata_path + 'total_movement_allPhaseroleWinseperation' + '_' + rankname + '.csv'
    pivot.to_csv(filename, index=False)
    # Step 4: Plotting the stacked bar graph
    fig, ax = plt.subplots()

    pivot.set_index(['Phase', 'individualPosition']).plot(kind='bar', y='total_avg_m', ax=ax, stacked=False, color='grey', alpha=0)

    # Add bars for w=0 and w=1 percentages
    for i, (index, row) in enumerate(pivot.iterrows()):
        ax.bar(i, row['total_avg_m'] * row['percentage_w0'], color='blue', label='w=0' if i == 0 else "")
        ax.bar(i, row['total_avg_m'] * row['percentage_w1'], bottom=row['total_avg_m'] * row['percentage_w0'],
               color='orange', label='w=1' if i == 0 else "")

    # Add labels and title
    ax.set_xticks(range(len(pivot)))
    ax.set_xticklabels([f"{Phase}-{role}" for Phase, role in zip(pivot['Phase'], pivot['individualPosition'])])
    ax.set_ylabel('Average m')
    ax.set_title('Average m per Phase and Role with w=0 and w=1 Contributions')
    ax.legend()

    plt.xticks(rotation=45)
    plt.show()

def total_movement_allPhaseRole(df,rankname):
    aggregated_data = df.groupby(['individualPosition', 'Phase'])['totalmovement distance per phase'].agg(['mean','std','count'])

    # Calculate SEM
    aggregated_data['sem'] = aggregated_data['std'] / np.sqrt(aggregated_data['count'])
    # Reshape the DataFrame using unstack
    aggregated_data = aggregated_data.unstack(fill_value=0)

    filename=outputdata_path+'total_movement_allPhaseRole' + '_'+ rankname+'.csv'
    aggregated_data.to_csv(filename)
    # Plot the bar graph
    fig, ax = plt.subplots(figsize=(10, 6))

    # Define positions and width
    positions = np.arange(len(aggregated_data))
    bar_width = 0.2

    # Plot each phase with different colors
    ax.bar(positions - bar_width, aggregated_data[1], width=bar_width, color='red', label='EarlyGame')
    ax.bar(positions, aggregated_data[2], width=bar_width, color='blue', label='MidGame')
    ax.bar(positions + bar_width, aggregated_data[3], width=bar_width, color='green', label='LateGame')

    # Set labels and title
    ax.set_xlabel('individualPosition')
    ax.set_ylabel('Average of total movement distance ')
    ax.set_title(f'Average of total movement distance for Each Role and Phase for {rankname} Players')
    ax.set_xticks(positions)
    ax.set_xticklabels(aggregated_data.index)
    ax.legend()

    # Show the plot
    plt.show()

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

inputdata_pathMT = "../data/OutputRank/MatchTimeline/MatchTimeline_WithExtraColumns.csv"

inputdata_pathRT="../data/OutputRank/MatchTimeline/RankedMatchTimeline_withExtraColumns.csv"


inputdata_pathMR = "../data/OutputRank/MatchTimeline/MatchTimeline_masterfile_PositionwithRole.csv"



inputdata_pathZone="../data/OutputRank/ClusterResults/ZoneperMinute_perParticipant.csv"

inputdata_pathMRank = "../data/OutputRank/MatchTimeline/MatchTimeline_masterfile_PositionwithRoleRank.csv"

inputdata_pathZoneRank = "../data/OutputRank/MatchTimeline/MatchTimeline_masterfile_PositionwithRoleRankall.csv"


#inputdata_pathTimePositionRank="../data/OutputRank/MatchTimeline/MatchTimeline_masterfile_PositionRowwithRoleRankPhase2024New.csv"
inputdata_pathTimePositionRank="../data/OutputRank/MatchTimeline/MatchTimeline_masterfile_PositionRowwithRoleRankPhase2024.csv"


outputdata_path = "../data/OutputRank/Spatio-temporal Analysis/"

phase_list=[1, 2, 3]
early_phaseTr=10 ##10 mins
mid_phaseTr=20 ##up to 20 mins

Role_list=['SOLO','JUNGLE','DUO','CARRY','SUPPORT']
color_list=['blue','orange','red','green','yellow','cyan','magenta']

output_file=outputdata_path  +'spatio-temporal clustering results.csv'
##needs to be run just one to make  to make ZoneperMinute_perParticipant
#find_numberofZoneChanged(inputdata_pathZone)
"""
df=pd.read_csv(inputdata_pathMR)
filtered_df = df[(df['participantId'] >= 1) & (df['participantId'] <= 5)]
dfg=filtered_df.groupby('Phase').agg({'totalmovement distance': ['mean','std'],'totalmovement distance per phase':'mean'})
print (dfg)
"""

### comparing different ranks of players for total movement with line plot
#calculate_TotalMovementandPlot(inputdata_pathTimePositionRank,outputdata_path)
#distance_statAnalysis(inputdata_pathMR,inputdata_pathMRank,inputdata_pathZoneRank)
####creating plots for movement vectors of players
#LICpatternMovement(inputdata_pathTimePositionRank)
###calculate new position based metrics such as split score, compainion metrics and rotation metrcis
callculate_movementRelatedScores(inputdata_pathTimePositionRank,outputdata_path)

"""
df=pd.read_csv(inputdata_pathRT)
df['individualPosition'] = df['individualPosition'].replace('NONE', 'JUNGLE')
df['lane'] = df['lane'].replace('NONE', 'JUNGLE')
df_phase = df[df['Phase'] == 1]
df_grouped=df.groupby('Rank').size().reset_index()
print (df_grouped)

metric=['damageStats_physicalDamageTaken','minionsKilled','championStats_healthMax','damageStats_magicDamageTaken']
for phase in phase_list:
    dfw=df[df['Rank']=='challenger']
    df_phase = dfw[dfw['Phase'] == phase]
    df_phase=df_phase[df_phase['individualPosition']=='JUNGLE']
   # df_phase = df_phase[df_phase['lane'] == 'TOP']
    X = df_phase[metric]
    df_results,k=ClusteringData(X,df_phase)
    Visualize_LolMapBasedonPosition(df_results,metric,phase,k)
"""

"""
#df=df[df['win']==0]
####clustering matchtimeline per different phase/role
for phase in phase_list:
    df_phase = df[df['Phase'] == phase]
    for role in Role_list:
        df_role=df_phase[df_phase['individualPosition']==role]
        X = df[['position_x', 'position_y']]
        df_results=ClusteringData(X,df_role)
        Visualize_LolMapBasedonPosition(df_results,role,phase)
"""
#KPI clustering FOR MATCHTIMELINE
####clustering matchtimeline per different phase (damageStats_physicalDamageTaken)
"""
metric=['damageStats_physicalDamageTaken','minionsKilled','championStats_healthMax','damageStats_magicDamageTaken']
for phase in phase_list:
    dfw=df[df['win']==0]
    df_phase = dfw[dfw['Phase'] == phase]
    df_phase=df_phase[df_phase['individualPosition']=='CARRY']
    df_phase = df_phase[df_phase['lane'] == 'TOP']
    X = df_phase[metric]
    df_results,k=ClusteringData(X,df_phase)
    Visualize_LolMapBasedonPosition(df_results,metric,phase,k)
"""