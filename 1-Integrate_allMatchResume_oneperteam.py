

import pandas as pd

import shutil
import os
import glob
from datetime import datetime
from difflib import SequenceMatcher
import time

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

def make_Avg(df):
    suffix = 'diff'
    df.replace({False: 0, True: 1}, inplace=True)
    # Calculate the average of numeric columns for the first 5 rows
    gameDuration=df.iloc[0]['gameDuration']
    #platformId=df.iloc[0]['platformId']
    timestamp_ms=df.iloc[0]['gameStartTimestamp']

    timestamp_seconds = timestamp_ms / 1000  # Convert milliseconds to seconds
    formatted_time = datetime.utcfromtimestamp(timestamp_seconds).strftime('%Y_%m_%d-%H_%M_%S')

    avg_first_5_rows = df.head(5).select_dtypes(include='number').mean()


    # Calculate the average of numeric columns for the next 5 rows
    avg_next_5_rows = df.iloc[5:10].select_dtypes(include='number').mean()
    avg_first_5_rows_diff = avg_first_5_rows - avg_next_5_rows

    avg_next_5_rows_diff=avg_next_5_rows-avg_first_5_rows




    # Rename the differences columns
    avg_first_5_rows_diff= avg_first_5_rows_diff.add_suffix(f'_{suffix}')

    avg_next_5_rows_diff= avg_next_5_rows_diff.add_suffix(f'_{suffix}')

    avg_first_5_rows_new=pd.concat([avg_first_5_rows,avg_first_5_rows_diff],axis=0)
    avg_next_5_rows_new = pd.concat([avg_next_5_rows, avg_next_5_rows_diff], axis=0)

    # Combine the averages into a new DataFrame
    result_df = pd.concat([avg_first_5_rows_new, avg_next_5_rows_new], axis=1).T
    result_df.insert(0, 'game_date', formatted_time)
    result_df.drop(['win_diff','gameStartTimestamp','gameStartTimestamp_diff','mapId_diff'], axis=1, inplace=True)
    return result_df

def make_Rolebaseddataset(df,Role_template):
    role_list=df['role']
    role_list.fillna('JUNGLE', inplace=True)
    role_list=["JUNGLE" if value == 'NONE' else value for value in role_list]
    role_list=["SUPPORT" if value == 'DUO' else value for value in role_list]

    #if not are_lists_match(role_list,Role_template):
     #   return (-1,None)
    df['role']=role_list
    grouped=df.groupby('role')

    separate_dfs = {group_name: group for group_name, group in grouped}
    return(0,separate_dfs)
def CreateOneRow_perTeam(df_cols):
    # Convert the DataFrame column to a Python array
    selected_array = df_cols.iloc[:, 0].tolist()
    csv_files = []
    for root, dirs, files in os.walk(inputMatchResume_path):
        for file in files:
            if file.endswith('.csv'):
                csv_files.append(os.path.join(root, file))

    ## Assuming all your CSV files are in the same directory
    # csv_files = glob.glob(inputMatchResume_path)

    # Initialize an empty DataFrame to store the data
    combined_df = pd.DataFrame()

    SOLO_df = pd.DataFrame()
    SUPPORT_df = pd.DataFrame()
    CARRY_df = pd.DataFrame()
    JUNGLE_df = pd.DataFrame()


    columnlist_notincluded = ['']

    # Initialize an empty list to store individual DataFrames
    dfs = []
    dfs_SOLO = []
    dfs_SUPPORT = []
    dfs_CARRY = []
    dfs_JUNGLE = []

    # Set the limit to the number of files you want to read (e.g., 10000)
    #files_limit = 5

    # Counter to keep track of the number of files processed
    files_processed = 0

    # Iterate through each CSV file and append its data to the combined DataFrame
    for csv_file in csv_files:
        # Read the CSV file into a DataFrame
        print(csv_file)
        df = pd.read_csv(csv_file)
        if len(df) > 0:
            # Select columns without an underscore
            if df['gameDuration'].iloc[0] >= 900:
                if set(selected_array).issubset(df.columns):
                    subset_df = df[selected_array]

                    ####1
                    """
                    ###for role based analysis
                    Is_match,multiple_df=make_Rolebaseddataset(subset_df, Role_template)
                    if Is_match==0:
                        dfs_SOLO.append(multiple_df['SOLO']) if 'SOLO' in multiple_df else dfs_SOLO
                        dfs_JUNGLE.append(multiple_df['JUNGLE']) if 'JUNGLE' in multiple_df else dfs_JUNGLE
                        dfs_SUPPORT.append(multiple_df['SUPPORT']) if 'SUPPORT' in multiple_df else dfs_SUPPORT
                        dfs_CARRY.append(multiple_df['CARRY']) if 'CARRY' in multiple_df else dfs_CARRY
                    """
                    ####2
                    current_df = make_Avg(subset_df)
                    # Append the current DataFrame to the list
                    dfs.append(current_df)

                    files_processed += 1
                    #if files_processed >= files_limit:
                    #    break
                    print(files_processed)
            # Increment the counter

        # Break the loop when the desired number of files is reached

    # Concatenate all DataFrames in the list into a single DataFrame
    combined_df = pd.concat(dfs, ignore_index=True)
    # get_lowVarienceCols(combined_df)
    combined_df.to_csv(outputMatchResume_path1, index=False)

    """
    SOLO_df = pd.concat(dfs_SOLO, ignore_index=True)
    SUPPORT_df = pd.concat(dfs_SUPPORT, ignore_index=True)
    CARRY_df = pd.concat(dfs_CARRY, ignore_index=True)
    JUNGLE_df = pd.concat(dfs_JUNGLE, ignore_index=True)

    SOLO_filename=outputdataRole_path+"Matchdata_SOLD.csv"
    SUPPORT_filename=outputdataRole_path+"Matchdata_SUPPORT.csv"
    CARRY_filename=outputdataRole_path+"Matchdata_CARRY.csv"
    JUNGLE_filename=outputdataRole_path+"Matchdata_JUNGLE.csv"

    SOLO_df.to_csv(SOLO_filename, index=False)
    SUPPORT_df.to_csv(SUPPORT_filename, index=False)
    CARRY_df.to_csv(CARRY_filename, index=False)
    JUNGLE_df.to_csv(JUNGLE_filename, index=False)
    """
    #
def are_lists_match(list1, list2):
    # Concatenate the lists into strings
    str1 = ' '.join(list1)
    str2 = ' '.join(list2)

    # Compute the similarity ratio between the two strings
    similarity_ratio = SequenceMatcher(None, str1, str2).ratio()

    # Check if the similarity ratio is at least 70%
    return similarity_ratio >= 0.7

#
###main
inputMatchResume_path = "../../RiotProject/data/OutputMatchResume1"#/*.csv"
outputMatchResume_path1 = "../data/OutputRank/MatchResume/MatchResume_Masterfile1.csv"
outputMatchResume_path2 = "../data/OutputRank/MatchResume/MatchResume_Masterfile2.csv"
outputMatchResume_path3 = "../data/OutputRank/MatchResume/MatchResume_TeamMasterfile.csv"

outputdataRole_path="../data/OutputRank/MatchResumePerRole/"

Role_template=['SOLO','JUNGLE','SOLO','CARRY','SUPPORT','SOLO','JUNGLE','SOLO','CARRY','SUPPORT']


inputdata_pathforSelectedCols = "../data/Input/MatchResume_SelectedCols.csv"
df_cols = pd.read_csv(inputdata_pathforSelectedCols)

#CreateOneRow_perTeam(df_cols)

df1 = pd.read_csv(outputMatchResume_path1)
df2 = pd.read_csv(outputMatchResume_path2)
df_total=pd.concat([df1,df2],axis=0,ignore_index=False)
df_group=df_total.groupby(['gameDuration']).agg({'gameDuration':['max','min','mean']}).reset_index()
print (df_group)

df_total.to_csv(outputMatchResume_path3, index=False)
