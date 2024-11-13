import pandas as pd
import numpy as np

import imageio
import pyvista as pv
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import ttest_ind

def correlation_analysisMatchTimeline(Inputdata_path2):
    df = pd.read_csv(Inputdata_path2)
    print(len(df))
    df = df.dropna(subset=['win'])
    print(len(df),'after removing null')
    df['win'] = df['win'].astype(int)
    df['role'] = df['role'].replace('NONE', 'JUNGLE')
    df['lane'] = df['lane'].replace('NONE', 'JUNGLE')

    for phase in phase_list:
        df_phase = df[df['time_interval'] == phase]
        df_cols=df.select_dtypes(include='number').columns
        df_phase=df_phase[df_cols]
        print (phase)
        correlation_analysis(df_phase)

def ttest_evaluation(df):
    # Specify the significance level (alpha) for the t-test
    alpha = 0.05

    # Store p-values for each feature
    p_values = {}
    # Perform t-test for each feature with the target variable
    significant_features = []
    for feature in df.columns[:-1]:  # Exclude the target variable
        feature_values_target_0 = df[df['win'] == 0][feature]
        feature_values_target_1 = df[df['win'] == 1][feature]

        # Perform t-test
        _, p_value = ttest_ind(feature_values_target_0, feature_values_target_1)

        # Store the p-value for the feature
        p_values[feature] = p_value

    # Sort p-values by value in ascending order
    sorted_p_values = sorted(p_values.items(), key=lambda x: x[1])

    # Display p-values for each feature
    for feature, p_value in sorted_p_values:
        print(f"{feature}: {p_value}")

    # Filter features based on significance level
    significant_features = [feature for feature, p_value in sorted_p_values if p_value > alpha]

    # Display significant features
    print("\nFeatures with p-value below threshold:")
    print(significant_features)

def correlation_analysis(df):
    # Calculate the correlation matrix
    df = df.dropna(subset=['win'])
    print(len(df),'after removing null')
    df['win'] = df['win'].astype(int)

    df_cols=df.select_dtypes('number').columns
    df=df[df_cols]
    correlation_matrix = df.corr()

    # Get the correlation of each feature with the target variable
    target_correlation = correlation_matrix['win'].drop('win').abs().sort_values(ascending=False)

    # Display the top correlated features
    top_correlated_features = target_correlation.index[:-20]
    print("Top 10 Correlated Features with Target Variable:")
    print(target_correlation[top_correlated_features])

def champion_analysis_winRate(Inputdata_path):
    df = pd.read_csv(Inputdata_path,nrows=10000)#, usecols=['time_interval', 'role', 'championName','win'])
    df.to_csv(onputdata_path)
    # Group by phase, role, and champion, then calculate win rate for each champion and role
    champion_role_stats = df.groupby(['time_interval', 'role', 'championName'])['win'].agg(['sum', 'count']).reset_index()
    champion_role_stats['win_rate'] = champion_role_stats['sum'] / champion_role_stats['count'] * 100

    # Sort by win rate within each phase and role
    sorted_champion_wins  = champion_role_stats.groupby(['time_interval', 'role']).apply(lambda x: x.nlargest(10, 'win_rate')).reset_index(drop=True)

      # Define colors for different phases
    phases = sorted_champion_wins['time_interval'].unique()
    phase_colors = ['blue','green','magenta']#,'plt.cm.tab10(np.linspace(0, 1, len(phases)))  # Use tab10 colormap for a set of distinct colors

    # Plot bar graphs for each phase
    for phase, color in zip(phases, phase_colors):
        phase_data = sorted_champion_wins[sorted_champion_wins['time_interval'] == phase]

        roles = phase_data['role'].unique()
        num_roles = len(roles)

        plt.figure(figsize=(8, 3 * num_roles))

        for i, role in enumerate(roles, start=1):
            plt.subplot(num_roles, 1, i)
            role_data = phase_data[phase_data['role'] == role]
            role_data_reverse = role_data.iloc[::-1]  # Reverse the DataFrame to have biggest bar at the top
            plt.barh(range(len(role_data_reverse)), role_data_reverse['win_rate'], height=0.05, color=color)
            plt.yticks(range(len(role_data_reverse)), role_data_reverse['championName'])
            plt.title(f'Top 10 Champions in {role} for {phase}')
            bars = plt.barh(role_data_reverse['championName'], role_data_reverse['win_rate'],
                 color=color)  # Bar plot for each role with respective color
            plt.ylabel('Champion')
            plt.xlabel('Win Rate')
            #plt.xlim(0, 1)  # Set the x-axis limit to show win rate from 0 to 100%
            #plt.xticks(rotation=45, ha='right')
            plt.xlim(0, 100)
            #plt.xticks(rotation=45, ha='right')
            plt.gca().xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{x/100:.0%}'))  # Add '%' symbol to tick labels
            plt.subplots_adjust(hspace=0.1)  # Adjust vertical spacing between subplots
            """for bar in bars:
                width  = bar.get_width()
                plt.text(width, bar.get_y() + bar.get_height() / 2, f'{width:.1f}%', ha='left', va='center')
            """
            plt.tight_layout()

        plt.show()

def champion_analysis_usedRate(Inputdata_path):
            df = pd.read_csv(Inputdata_path, usecols=['time_interval', 'role', 'championName', 'win'])
            # Group by phase, role, and champion, then calculate win rate
            win_rate = df.groupby(['time_interval', 'role', 'championName'])['win'].size().reset_index()

            # Sort by win rate within each phase and role
            win_rate_sorted = win_rate.groupby(['time_interval', 'role']).apply(
                lambda x: x.nlargest(10, 'win')).reset_index(drop=True)

            # Define colors for different phases
            phases = win_rate_sorted['time_interval'].unique()
            phase_colors = ['blue', 'green',
                            'magenta']  # ,'plt.cm.tab10(np.linspace(0, 1, len(phases)))  # Use tab10 colormap for a set of distinct colors

            # Find maximum win rate among all phases and roles
            max_win_rate = win_rate['win'].max()

            # Plot bar graphs for each phase

            for phase, color in zip(phases, phase_colors):
                phase_data = win_rate_sorted[win_rate_sorted['time_interval'] == phase]
                phase_datalen = len(phase_data)
                roles = phase_data['role'].unique()
                num_roles = len(roles)

                plt.figure(figsize=(5, 2 * num_roles))

                for i, role in enumerate(roles, start=1):
                    plt.subplot(num_roles, 1, i)
                    role_data = phase_data[phase_data['role'] == role]
                    role_data_reverse = role_data.iloc[::-1]  # Reverse the DataFrame to have biggest bar at the top
                    plt.barh(range(len(role_data_reverse)), role_data_reverse['win'], height=0.3, color=color)
                    plt.yticks(range(len(role_data_reverse)), role_data_reverse['championName'])
                    plt.title(f'Top 10 Champions in {role} for {phase}')
                    plt.xlabel('Champion')
                    plt.ylabel('Win Rate')
                    # plt.xlim(0, 1)  # Set the x-axis limit to show win rate from 0 to 100%
                    # plt.xticks(rotation=45, ha='right')
                    plt.xlim(0, max_win_rate + 10)
                    # x_ticks = np.arange(0, 100, 20)
                    # plt.xticks(x_ticks)
                    # plt.xticks(np.arange(0, max_win_rate + 0.1, 0.1))  # Set x-ticks every 0.1
                    # plt.gca().xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{x/100:.0%}'))  # Add '%' symbol to tick labels
                    plt.subplots_adjust(hspace=0.1)  # Adjust vertical spacing between subplots
                    plt.tight_layout()

                plt.show()

def champion_analysis_winRateperLane(Inputdata_path):
    df = pd.read_csv(Inputdata_path, nrows=100000, usecols=['time_interval', 'role', 'lane','championName','win'])
    # Group by phase, role, and champion, then calculate win rate for each champion and role
    champion_role_stats = df.groupby(['time_interval', 'role','lane', 'championName'])['win'].agg(['sum', 'count']).reset_index()
    champion_role_stats['win_rate'] = champion_role_stats['sum'] / champion_role_stats['count'] * 100

    # Sort by win rate within each phase and role
    sorted_champion_wins  = champion_role_stats.groupby(['time_interval', 'role','lane']).apply(lambda x: x.nlargest(10, 'win_rate')).reset_index(drop=True)

    print(sorted_champion_wins)

    champion_win_stats = df.groupby(['time_interval','championName', 'role', 'lane'])['win'].agg(['sum']).reset_index()
    champion_optimal_stats = champion_win_stats.groupby('championName').apply(
        lambda x: x.nlargest(1, 'sum')).reset_index(drop=True)

    print(champion_optimal_stats)

    # Sort data per role, lane, and sum of wins
    sorted_champion_optimal_stats = champion_win_stats.sort_values(by=['time_interval','role', 'lane', 'sum'],
                                                                   ascending=[True, True, False])

    print(sorted_champion_optimal_stats)

      # Define colors for different phases
    phases = sorted_champion_wins['time_interval'].unique()
    phase_colors = ['blue','green','magenta']#,'plt.cm.tab10(np.linspace(0, 1, len(phases)))  # Use tab10 colormap for a set of distinct colors

    # Plot bar graphs for each phase
    for phase, color in zip(phases, phase_colors):
        phase_data = sorted_champion_wins[sorted_champion_wins['time_interval'] == phase]

        roles = phase_data['role'].unique()
        lanes = phase_data['lane'].unique()
        num_roles = len(roles)

        plt.figure(figsize=(8, 3 * num_roles))
        plt.suptitle(phase, fontsize=16, color=color)
        for i, role in enumerate(roles, start=1):
            for j, lane in enumerate(lanes, start=1):
                plt.subplot(num_roles, len(lanes), (i - 1) * len(lanes) + j)

                role_lane_data = phase_data[(phase_data['role'] == role) & (phase_data['lane'] == lane)]

                role_data_reverse = role_lane_data.iloc[::-1]


                plt.barh(range(len(role_data_reverse)), role_data_reverse['win_rate'], height=0.05, color=color)
                plt.yticks(range(len(role_data_reverse)), role_data_reverse['championName'])
                plt.ylabel('Champion')
                plt.title(f'Top 10 Champions in {role} - {lane} -{phase}')
                plt.xlabel('Win Rate (%)')
                #plt.xlim(0, 1)  # Set the x-axis limit to show win rate from 0 to 100%
                #plt.xticks(rotation=45, ha='right')
                plt.xlim(0, 100)
                #plt.xticks(rotation=45, ha='right')
                plt.gca().xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{x/100:.0%}'))  # Add '%' symbol to tick labels
                plt.subplots_adjust(hspace=0.1)  # Adjust vertical spacing between subplots
                """for bar in bars:
                    width  = bar.get_width()
                    plt.text(width, bar.get_y() + bar.get_height() / 2, f'{width:.1f}%', ha='left', va='center')
                """
                plt.tight_layout()

        plt.show()

def get_matchchallengesRolestats(Inputdata_path3,onputdata_path):
    df = pd.read_csv(Inputdata_path3, usecols=['matchId', 'role', 'lane'])
    df['role'] = df['role'].replace('NONE', 'JUNGLE')
    df['lane'] = df['lane'].replace('NONE', 'JUNGLE')

    role_percentage = df['role'].value_counts(normalize=True) * 100
    # Write the first DataFrame to a CSV file
    with open(onputdata_path, 'a') as f:
        f.write("Distribution of different roles in match dataset \n")  # Write title
    role_percentage.to_csv(onputdata_path, index=True)

    lane_role_percentage = df.groupby('lane')['role'].value_counts(normalize=True) * 100
    # Append the second DataFrame to the same CSV file
    with open(onputdata_path, 'a') as f:
        f.write("Distribution of different roles and lanes in match dataset \n")  # Write title
    lane_role_percentage.to_csv(onputdata_path, mode='a', index=True, header=True)







   # plotter.set_background('white')

    # Show the plot
   # plotter.show()



Inputdata_path2 = "data/OutputRank/MatchTimeline/MatchTimeline_PerMatchPhase.csv"
Inputdata_path3 = "data/OutputRank/MatchTimeline/MatchTimeline_PerMatchParticipant.csv"

onputdata_path = "data/OutputRank/Analysis/MatchResume_RoleStats.csv"
phase_list=['EarlyPhase', 'MidPhase', 'LatePhase']
#df = pd.read_csv(Inputdata_path1)

#get_matchchallengesRolestats(Inputdata_path3,onputdata_path)
champion_analysis_usedRate(Inputdata_path2)

#champion_analysis_winRate(Inputdata_path2)

#champion_analysis_winRateperLane(Inputdata_path2)

#correlation_analysis(df)
#correlation_analysisMatchTimeline(Inputdata_path2)
#visualization_analysis()