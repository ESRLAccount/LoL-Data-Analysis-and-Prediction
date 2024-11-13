
"""
Created on Tus 20 March 2024

#project: LOL KPI analysis

creating files by reading each of matchtimeline and then structureing data by calculating mean of metrics per partciapnts per team
@author: Fazilat
"""
import pandas as pd
import numpy as np
import shutil
import os
import glob
from datetime import datetime
import time
import warnings
import re

def remove_digit_prefix(df):

    # Regular expression pattern to match a digit followed by an underscore at the beginning
    pattern = re.compile(r'^\d+_')


    # Create a dictionary to map old column names to new column names
    new_columns = {col: re.sub(pattern, '', col) for col in df.columns}

    # Rename the columns
    df.rename(columns=new_columns, inplace=True)
    return (df)

# Function to remove the leading digit from column names
def remove_leading_digit(column_name):
    if column_name[0].isdigit():
        return column_name[2:]
    else:
        return column_name

def make_avgperteamphase(df):
    # Calculate the average of rows where A is between 1 and 5
    df_team1 = df[df['participant_number'].between(1, 5)]

    df_team1=df_team1.groupby('phase').mean().reset_index()

    # Calculate the average of rows where A is greater than 5
    df_team2 = df[df['participant_number'] > 5]

    df_team2=df_team2.groupby('phase').mean().reset_index()

    df_team1['team_id']=1
    df_team2['team_id'] = 2
    concatenated_df = pd.concat([df_team1, df_team2], axis=0)
    concatenated_df.drop(['participant_number'],axis=1,inplace=True)
    return (concatenated_df)


def last_valid_or_second_last(series):
    last_valid_index = series.last_valid_index()
    last_value = series.iloc[-1]

    if pd.isna(last_value) :
        return series.iloc[-2]
    return last_value



def calculate_Player_distancemetrics(df):
    df = df.reset_index(drop=False)

    dx = df['position_x'].diff(periods=-1)
    dy = df['position_y'].diff(periods=-1)

    df['xposition_diff']=dx
    df['yposition_diff']=dy

    # Calculate the distance using the Pythagorean theorem
    distances = np.sqrt(dx ** 2 + dy ** 2)
    distances.iloc[0] = 0
    df['movement distance']=distances
    df['totalmovement distance']=distances.cumsum()

    df.rename(columns={'index': 'row_number'}, inplace=True)

    conditions = [df['row_number'] < early_phaseTr,
                  (df['row_number'] >= early_phaseTr) & (df['row_number'] < mid_phaseTr),
                  df['row_number'] >= mid_phaseTr]
    choices = ['1', '2', '3']  ### 1: Early Phase, 2: Mid Phase, 3: late phase
    df['Phase'] = np.select(conditions, choices, default='1')
    df_grouped = df.groupby('Phase')

    result = df_grouped.agg({col: 'mean' for col in cols_for_Avg} | {col: 'last' for col in cols_for_Last}
                            | {col: 'sum' for col in cols_for_sum})# | { 'totalmovement distance': lambda x: last_valid_or_second_last(x) - x.iloc[0]})  # Last value minus first value ).reset_index(drop=False )

    result['totalmovement distance per phase'] = result['totalmovement distance'].diff(periods=1)
    result['totalmovement distance per phase'].fillna(result['totalmovement distance'], inplace=True)
    #result['totalmovement distance2'] = result.apply(lambda row: row['totalmovement distance'] if row['totalmovement distance2'] == '' else row['totalmovement distance2'], axis=1)
    return (result)



def make_avgpergamephase(df):
     df = df.reset_index(drop=False)
     df.rename(columns={'index': 'row_number'}, inplace=True)

     conditions = [df['row_number'] < early_phaseTr,
                  (df['row_number'] >= early_phaseTr) & (df['row_number'] < mid_phaseTr),
                  df['row_number'] >= mid_phaseTr]
     choices = ['1', '2', '3'] ### 1: Early Phase, 2: Mid Phase, 3: late phase
     df['Phase'] = np.select(conditions, choices, default='1')
     df_grouped=df.groupby('Phase')
     result = df_grouped.agg({col: 'mean' for col in cols_for_Avg} | {col: 'last' for col in cols_for_Last}).reset_index(drop=False)


     return (result)


def calculate_team_distancemetrics(df):
    # Calculate the distance for the first five x and y columns
    first_five_x_cols = [f'{i}_position_x' for i in range(1, 6)]
    first_five_y_cols = [f'{i}_position_y' for i in range(1, 6)]

    first_five_distances = []

    for x_col, y_col in zip(first_five_x_cols, first_five_y_cols):
        dx = df[x_col].diff()
        dy = df[y_col].diff()
        dx.iloc[0] = 0  # Fill the first row with 0
        dy.iloc[0] = 0
        distances = np.sqrt(dx ** 2 + dy ** 2)
        first_five_distances.append(distances.sum())

    total_distance_first_five = sum(first_five_distances)

    # Calculate the distance for the second five x and y columns
    second_five_x_cols = [f'{i}_position_x' for i in range(6, 11)]
    second_five_y_cols = [f'{i}_position_y' for i in range(6, 11)]

    second_five_distances = []

    for x_col, y_col in zip(second_five_x_cols, second_five_y_cols):
        dx = df[x_col].diff()
        dy = df[y_col].diff()
        dx.iloc[0] = 0  # Fill the first row with 0
        dy.iloc[0] = 0
        distances = np.sqrt(dx ** 2 + dy ** 2)
        second_five_distances.append(distances.sum())

    total_distance_second_five = sum(second_five_distances)

    return (total_distance_first_five,total_distance_second_five)


def CreateVerticalDataframe(df,cols_name,matchId):
    participant_numbers = set(int(column.split('_')[0]) for column in df.columns if column[0].isdigit())

    td_team1,td_team2=calculate_team_distancemetrics(df)

    # Extract and concatenate each participant's data
    participant_data_list = []

    for participant_number in participant_numbers:
        # Select columns with the current participant number
        selected_columns = [column for column in df.columns if column.startswith(f'{participant_number}_')]

        # Create a new DataFrame with selected columns
        participant_data = df[selected_columns]

        # Append the participant's data to the list
        # Move the last column(s) to the position of the first column(s)
        participant_data = pd.concat([participant_data.iloc[:, -1:], participant_data.iloc[:, :-1]], axis=1)


        participant_data=remove_digit_prefix(participant_data)


        #df_result=make_avgpergamephase(participant_data,td_team1,td_team2)

        df_result=calculate_Player_distancemetrics(participant_data)

        df_result['participantId']=participant_number
        participant_data_list.append(df_result)

def CreateVerticalDataframe_Position(df, cols_name, matchId):
        participant_numbers = set(int(column.split('_')[0]) for column in df.columns if column[0].isdigit())

        td_team1, td_team2 = calculate_team_distancemetrics(df)

        # Extract and concatenate each participant's data
        participant_data_list = []

        for participant_number in participant_numbers:
            # Select columns with the current participant number
            selected_columns = [column for column in df.columns if column.startswith(f'{participant_number}_')]

            # Create a new DataFrame with selected columns
            participant_data = df[selected_columns]

            # Append the participant's data to the list
            # Move the last column(s) to the position of the first column(s)
            participant_data = pd.concat([participant_data.iloc[:, -1:], participant_data.iloc[:, :-1]], axis=1)

            participant_data = remove_digit_prefix(participant_data)

            # df_result=make_avgpergamephase(participant_data,td_team1,td_team2)

            df_result = calculate_Player_distancemetrics(participant_data)

            df_result['participantId'] = participant_number
            participant_data_list.append(df_result)

        # Concatenate the participant data in a row-wise manner
        participant_data = pd.concat(participant_data_list, axis=0, ignore_index=True)

        # Get the mean for columns in the list, and last value for columns not in the list

        participant_data['total_movement_team1'] = td_team1
        participant_data['total_movement_team2'] =td_team2

        #participant_data=make_avgperteamphase(participant_data)


        participant_data['match_id'] = matchId

        return (participant_data)


def CreateVerticalDataframe_position(df,cols_name,matchId):
    participant_numbers = set(int(column.split('_')[0]) for column in df.columns if column[0].isdigit())


    # Extract and concatenate each participant's data
    participant_data_list = []

    for participant_number in participant_numbers:
        # Select columns with the current participant number
        selected_columns = [column for column in df.columns if column.startswith(f'{participant_number}_')]

        # Create a new DataFrame with selected columns
        participant_data = df[selected_columns]

       # participant_data['participantId'] = participant_number
        # Append the participant's data to the list
        # Move the last column(s) to the position of the first column(s)
        participant_data = pd.concat([participant_data.iloc[:, -1:], participant_data.iloc[:, :-1]], axis=1)



        participant_data.columns = participant_data.columns.str.replace(r'^(\d+)_', '', regex=True)

        #df_result=make_avgpergamephase(participant_data,td_team1,td_team2)

       # participant_data['participant_number'] = participant_number

        participant_data=participant_data[cols_name]

        participant_data_list.append(participant_data)


    # Concatenate the participant data in a row-wise manner
    participant_data = pd.concat(participant_data_list, axis=0, ignore_index=True)

    # Get the mean for columns in the list, and last value for columns not in the list


    #participant_data=make_avgperteamphase(participant_data)


    participant_data['match_id'] = matchId

    return (participant_data)


def get_lowVarienceCols(df):
    # Calculate the variance for each column


    numeric_columns = combined_df.select_dtypes(include='number')

    variances = numeric_columns.var()

    # Set a threshold to identify columns with low variance
    variance_threshold = 0.01  # Adjust this threshold based on your needs

    # Get the columns with variance below the threshold
    low_variance_columns = variances[variances < variance_threshold].index

    # Print or use the low variance columns
    print("Columns with low variance:")
    print(low_variance_columns)
    return(low_variance_columns)

def make_Avg(df):
    df.replace({False: 0, True: 1}, inplace=True)
    # Calculate the average of numeric columns for the first 5 rows
    gameDuration=df.iloc[0]['gameDuration']
    platformId=df.iloc[0]['platformId']
    timestamp_ms=df.iloc[0]['gameStartTimestamp']

    timestamp_seconds = timestamp_ms / 1000  # Convert milliseconds to seconds
    formatted_time = datetime.utcfromtimestamp(timestamp_seconds).strftime('%Y_%m_%d-%H_%M_%S')

    avg_first_5_rows = df.head(5).select_dtypes(include='number').mean()

    # Calculate the average of numeric columns for the next 5 rows
    avg_next_5_rows = df.iloc[5:10].select_dtypes(include='number').mean()

    # Combine the averages into a new DataFrame
    result_df = pd.concat([avg_first_5_rows, avg_next_5_rows], axis=1).T
    # Insert the new columns at the beginning of the DataFrame
    result_df.insert(0, 'game_date', formatted_time)
    result_df.insert(1, 'platform', platformId)
   # result_df.insert(2, 'game_duration', gameDuration)
    return result_df


###main


# Disable all warnings
warnings.filterwarnings("ignore")

early_phaseTr=10 ##10 mins
mid_phaseTr=20 ##up to 20 mins

###these columns are used for making average in matchtimeline, the rest columns use Last function
cols_for_Avg=['championStats_abilityPower',
'championStats_attackSpeed',
'championStats_ccReduction',
'championStats_health',
'championStats_healthRegen',
'championStats_movementSpeed',
'championStats_power',
'championStats_powerRegen',
'currentGold',
'position_x',
'position_y'
]

cols_for_sum=['xposition_diff','yposition_diff']


inputdata_path = "../RiotProject/data/OutputMatchTimeline2"#/*.csv"
#inputdata_path = "../../LOLDataset/MatchTimeline_part2/*.csv"
outputdata_path = "../data/OutputRank/MatchTimeline/"

outputfilename=outputdata_path + 'MatchTimeline_PerPhaseMean_part2.csv'

outputfilename_merge=outputdata_path + 'MatchTimeline_PerPhaseMeanSampleMerge.csv'


Inputdata_pathforMasterfile = "data/Output/MasterFile/Masterfile_RowScore.csv"

#df_masterfile = pd.read_csv(Inputdata_pathforMasterfile,usecols=['win','matchId'])

inputdata_pathforSelectedCols = "data/Input/MatchTimeline_FinalCols.csv"
df_cols = pd.read_csv(inputdata_pathforSelectedCols)

# Convert the DataFrame column to a Python array
cols_name = df_cols.iloc[:, 0].tolist()
for i in range(len(cols_name)):
    cols_name[i]=remove_leading_digit(cols_name[i])

    # Convert lists to sets
set1 = set(cols_for_Avg)
set2 = set(cols_for_sum)

union_set = set1.union(set2)
cols_for_Last=[col for col in cols_name if col not in union_set ]
cols_for_Last.append('totalmovement distance')

colsname_forpositiondf=['participantId','position_x','position_y']
## Assuming all your CSV files are in the same directory
#csv_files = glob.glob(inputdata_path)
csv_files=[]
for root, dirs, files in os.walk(inputdata_path):
    for file in files:
        if file.endswith('.csv'):
            csv_files.append(os.path.join(root, file))

# Initialize an empty DataFrame to store the data
combined_df = pd.DataFrame()


columnlist_notincluded=['']

# Initialize an empty list to store individual DataFrames
dfs = []
# Set the limit to the number of files you want to read (e.g., 10000)
#files_limit =1000

# Counter to keep track of the number of files processed
files_processed = 0

# Iterate through each CSV file and append its data to the combined DataFrame
for csv_file in csv_files:
    # Read the CSV file into a DataFrame
    print (csv_file, files_processed)
    df = pd.read_csv(csv_file)
    matchId = df.iloc[0, -1]
    df.drop('matchId', axis=1, inplace=True)
    df = df.iloc[:, 1:]
    #if files_processed>files_limit:
    #    break;
    files_processed=files_processed+1
    print(files_processed)
    column_starts_with_10_ = any(col.startswith("10_") for col in df.columns)

    if column_starts_with_10_:
        #current_df=CreateVerticalDataframe(df, cols_name, matchId)
        current_df=CreateVerticalDataframe_position(df,colsname_forpositiondf, matchId)
        dfs.append(current_df)




outputfilename=outputdata_path + 'MatchTimeline_masterfile_positionRowdata2.csv'
# Concatenate all DataFrames in the list into a single DataFrame
combined_df = pd.concat(dfs, ignore_index=False)
combined_df = combined_df.drop_duplicates()
combined_df.to_csv(outputfilename, index=True)



#low_variance_columns=get_lowVarienceCols(combined_df)

#final_df = combined_df.drop(columns=low_variance_columns, axis=1)

#final_df.to_csv(outputfilename, index=False)

#df_masterfile['team_id']=df_masterfile.index%2+1

#merge_df=pd.merge(final_df, df_masterfile, how='left', on='team_id')

#merge_df.to_csv(outputfilename_merge,index=False)