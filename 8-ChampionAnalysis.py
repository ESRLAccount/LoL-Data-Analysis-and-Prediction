####
#
"""
Created on  Apr 11 14:40:32 2024
Understanding the optimal roles/positions for each champion in different game phases
@author: Fazilat
"""

import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
from scipy.stats import ttest_ind

def Normalize_perGameduration(df,fieldname):
    # Define the column names to exclude from normalization
    exclude_cols = ['win']  # Add any column names you want to exclude
    numericCols = df.select_dtypes(include='number').columns

    df=df[numericCols]

    df = df.apply(pd.to_numeric, errors='coerce')

    # df['gameDuration'] = pd.to_numeric(df['gameDuration'], errors='coerce')

    # Get the list of columns to normalize (all columns except exclude_cols)
    normalize_cols = df.columns.difference(exclude_cols)

    normalized_df = df.copy()

    # Normalize all columns based on the game duration
    normalized_df[normalize_cols] = normalized_df[normalize_cols].div(normalized_df[fieldname], axis=0)

    return (normalized_df)

def rank_Champions_perPhase(df,metric_scores_path,outputdata_path):
    included_cols=['role','lane','Phase','championName']
    for phase in phase_list:
        df_phase=df[df['Phase']==phase]
        for role in Role_list:
            metric_rolefile =metric_scores_path+role+'.csv'
            weightScore=calculate_WeightScore(metric_rolefile)
            df_role = df_phase[df_phase['role'] == role]
            df_stringCols=df_role[included_cols]

            df_normalized = Normalize_perGameduration(df_role, 'gameDuration')
            df_role=pd.concat([df_stringCols,df_normalized],axis=1)

            composit_score=0
            for index, row in weightScore.iterrows():
                name = row['Feature']
                score = row['Weight']
                composit_score=df_role[name]*score + composit_score

            df_role['Composite_Score']=composit_score

            # Group by champion name and calculate average composite score
            avg_scores = df_role.groupby('championName').agg({
                'Composite_Score': 'mean',  # Calculate mean of composite score
                'role': 'first' ,           # Take the first region value
                'lane':'first',
                'Phase':'first'
            }).reset_index()


            # Sort DataFrame by average composite score in descending order
            avg_scores_sorted = avg_scores.sort_values(by='Composite_Score', ascending=False)

            top_champions = avg_scores_sorted[['lane','championName','Composite_Score']].head(5)

            print(top_champions)

            with open(outputdata_path, 'a') as f:
                f.write(f'Top 10 Champions in {phase} for {role}')  # Write title
            top_champions.to_csv(outputdata_path, mode='a',index=True)


            # Define colors for each lane
            colors = ['skyblue', 'orange', 'green', 'red', 'purple']

            # Create the grouped bar chart
            plt.figure(figsize=(10, 6))
            for i, lane in enumerate(lanes):
                plt.bar(x + i*0.15, scores[i], width=0.15, label=lane, color=colors[i])

            # Add labels and title
            plt.xlabel('Champion')
            plt.ylabel('Average Composite Score')
            plt.title('Top Champions Based on Average Composite Score and Lane')
            plt.xticks(x + 0.3, champions, rotation=45, ha='right')  # Adjust x-axis ticks for better readability
            plt.legend()

            # Show the plot
            plt.tight_layout()
            plt.show()
################################
def rank_Champions_perRole(df, metric_scores_path, outputdata_path):
        included_cols = ['role', 'lane',  'championName']
        for role in Role_list:
                metric_rolefile = metric_scores_path + role + '.csv'
                weightScore = calculate_WeightScore(metric_rolefile)
                df_role = df[df['role'] == role]
                df_stringCols = df_role[included_cols]

                df_normalized = Normalize_perGameduration(df_role, 'gameDuration')
                df_role = pd.concat([df_stringCols, df_normalized], axis=1)

                composit_score = 0
                for index, row in weightScore.iterrows():
                    name = row['Feature']
                    score = row['Weight']
                    composit_score = df_role[name] * score + composit_score

                df_role['Composite_Score'] = composit_score

                # Group by champion name and calculate average composite score
                avg_scores = df_role.groupby('championName').agg({
                    'Composite_Score': 'mean',  # Calculate mean of composite score
                    'role': 'first',  # Take the first region value
                    'lane': 'first'

                }).reset_index()

                # Sort DataFrame by average composite score in descending order
                avg_scores_sorted = avg_scores.sort_values(by='Composite_Score', ascending=False)

                top_champions = avg_scores_sorted[['lane', 'championName', 'Composite_Score']].head(5)

                print(top_champions)

                with open(outputdata_path, 'a') as f:
                    f.write(f'Top 10 Champions  for {role}')  # Write title
                top_champions.to_csv(outputdata_path, mode='a', index=True)

                # Define colors for each lane
                colors = ['skyblue', 'orange', 'green', 'red', 'purple']

                # Create the grouped bar chart
                plt.figure(figsize=(10, 6))
                """for i, lane in enumerate(lanes):
                    plt.bar(x + i * 0.15, scores[i], width=0.15, label=lane, color=colors[i])

                # Add labels and title
                plt.xlabel('Champion')
                plt.ylabel('Average Composite Score')
                plt.title('Top Champions Based on Average Composite Score and Lane')
                plt.xticks(x + 0.3, champions, rotation=45,
                           ha='right')  # Adjust x-axis ticks for better readability
                plt.legend()

                # Show the plot
                plt.tight_layout()
                plt.show()"""


def calculate_WeightScore(metric_filepath):

        df_score = pd.read_csv(metric_filepath)
        df_score['SHAP Score']=abs(df_score['SHAP Score'])

        # Normalize scores
        df_score['Normalized_Score'] = (df_score['SHAP Score'] - df_score['SHAP Score'].min()) / (df_score['SHAP Score'].max() - df_score['SHAP Score'].min())

        # Define weights proportional to normalized scores
        df_score['Weight'] = df_score['Normalized_Score'] / df_score['Normalized_Score'].sum()

        # Ensure weights sum to 1
        total_weight = df_score['Weight'].sum()
        df_score['Weight'] /= total_weight

        return (df_score)





Inputdata_path1 = "data/OutputRank/MatchTimeline/MatchTimeline_PerMatchPhase.csv"
Inputdata_path_role = "data/OutputRank/MatchTimeline/MatchTimeline_PerMatchParticipant.csv"

metric_scores_path="data/OutputRank/FeatureSelection/KPI_MatchTimelinePerRole_"


outputdata_path = "data/OutputRank/Analysis/TopChampions_perRoleMatchPhase.csv"

outputdata_path_role = "data/OutputRank/Analysis/TopChampions_perRole.csv"

phase_list=['EarlyPhase', 'MidPhase', 'LatePhase']
Role_list=['SOLO','JUNGLE','DUO','CARRY','SUPPORT']

df = pd.read_csv(Inputdata_path_role)
print(len(df))
df = df.dropna(subset=['win'])
print(len(df), 'after removing null')
df['win'] = df['win'].astype(int)
df['role'] = df['role'].replace('NONE', 'JUNGLE')
df['lane'] = df['lane'].replace('NONE', 'JUNGLE')

rank_Champions_perRole(df,metric_scores_path,outputdata_path)

#rank_Champions_perPhase(df,metric_scores_path,outputdata_path_role)
