
"""
Created on Tus 2 Apr 2024

#project: LOL KPI analysis

1- creating a masterfile by reading data of all 3 summary files, then joining with masterfile_challenge to include win/loss for each row
2- creating a masterfile including all data from all matchtimeline files
@author: Fazilat
"""
import pandas as pd

import shutil
import os
import glob
from datetime import datetime
import time


def filter_2024Data(outputdata_pathRFinal, outputdata_path2024Final):

    df = pd.read_csv(outputdata_pathRFinal)

    # Ensure the 'date_column' is in datetime format
    df['year'] = df['game_date'].str[:4]

    # Filter rows where the year is '2023'
    rows_starting_with_2024 = df[df['year'] == '2024']

    # Get the number of such rows
    num_rows = len(rows_starting_with_2024)

    print(num_rows)

    print(df.columns)

    rows_starting_with_2024.to_csv(outputdata_path2024Final, index=True)

def filter_2024MatchData(file1, file2):

    df = pd.read_csv(file1)

    rows_starting_with_2024 = df[~df['game_date'].str.startswith('2023')]

    # Filter rows where the year is '2023'
    #rows_starting_with_2024 = df[df['matchId'] > 'EUW1_6745998743']

    # Get the number of such rows
    num_rows = len(rows_starting_with_2024)

    print(num_rows)

    rows_starting_with_2024.to_csv(file2, index=True)

def CreateVerticalDataframe(df,index):
    participant_numbers = set(int(column.split('_')[0]) for column in df.columns if column[0].isdigit())

    # Extract and concatenate each participant's data
    participant_data_list = []
    matchId = df['matchId']
    time=df['time']
    for participant_number in participant_numbers:
        # Select columns with the current participant number
        selected_columns = [column for column in df.columns if column.startswith(f'{participant_number}_')]

        # Create a new DataFrame with selected columns
        participant_data = df[selected_columns]


        # Remove the participant number from column names
        participant_data.columns =[column.split('_',1)[index] for column in selected_columns]

        participant_data['participantId'] = participant_number
        # Append the participant's data to the list
        # Move the last column(s) to the position of the first column(s)
        participant_data = pd.concat([participant_data.iloc[:, -1:], participant_data.iloc[:, :-1],matchId,time], axis=1)



        participant_data_list.append(participant_data)

    # Concatenate the participant data in a row-wise manner
    concatenated_data = pd.concat(participant_data_list, axis=0, ignore_index=True)

    return (concatenated_data)



def read_and_merge_hugedatafiles(folder_path):
    csv_files = glob.glob(folder_path)
    file1 = csv_files[0]
    file2 = csv_files[1]

    # Define chunk size (adjust according to your system's memory)
    chunk_size = 200000  # 1 million rows

    # Initialize an empty DataFrame to store the merged data
    merged_df = pd.DataFrame()
    dfs=[]
    i=0
    # Iterate over chunks from the first CSV file
    for chunk1, chunk2 in zip(pd.read_csv(file1, chunksize=chunk_size),
                              pd.read_csv(file2, chunksize=chunk_size)):
        # Merge or concatenate chunks
        dfs.append(pd.concat([chunk1, chunk2], ignore_index=True))
        print (i)
        i=i+1
    merged_df = pd.concat(dfs, ignore_index=True)
    return(merged_df )

def read_and_merge_dataframes(folder_path):
    # Get a list of all CSV files in the folder

    csv_files = glob.glob(folder_path)

    # Initialize an empty list to store DataFrames
    dfs = []
    file_processed=1
    # Loop through each CSV file and read it into a DataFrame
    for file in csv_files:
        #file_path = os.path.join(folder_path, file)
        df = pd.read_csv(file)
        dfs.append(df)
        file_processed=file_processed+1
        if file_processed>10:
            break;
    # Concatenate all DataFrames in the list into a single DataFrame
    combined_df = pd.concat(dfs, ignore_index=True)

    return combined_df

def read_and_merge_twofiles(file1,file2):
    # Get a list of all CSV files in the folder


        #file_path = os.path.join(folder_path, file)
        df1 = pd.read_csv(file1)
        df2 = pd.read_csv(file2)

        # Concatenate all DataFrames in the list into a single DataFrame
        combined_df = pd.concat([df1,df2], ignore_index=True)

        return combined_df


def read_and_merge_datafiles(folder_path):
    # Get a list of all CSV files in the folder

    csv_files = glob.glob(folder_path)

    # Initialize an empty list to store DataFrames
    dfs = []
    file_processed=1
    # Loop through each CSV file and read it into a DataFrame
    for file in csv_files:
        #file_path = os.path.join(folder_path, file)
        df = pd.read_csv(file)
        ##if the game duration is less than 15 mins, we exclude that
        if len(df)>=15:
           df['time'] = range(1, len(df) + 1)
           dfs.append(df)
        file_processed=file_processed+1
        print (file_processed)
        #if file_processed>10:
        #    break;
    # Concatenate all DataFrames in the list into a single DataFrame
    combined_df = pd.concat(dfs, ignore_index=True)

    return combined_df

#####
def read_and_join_two_huge_dataframe(file1, file2):



    # Step 1: Read both CSV files in chunks
    chunksize = 1000000  # Adjust chunk size based on available memory
    df1_chunks = pd.read_csv(file1, chunksize=chunksize)

    df2_chunks = pd.read_csv(file2, chunksize=chunksize)

    # Step 2: Initialize an empty list to store DataFrames
    dfs = []
    i=0

    # Step 3: Iterate over chunks from both DataFrames
    for df1_chunk, df2_chunk in zip(df1_chunks, df2_chunks):
        # Step 3a: Select only necessary columns
        df1_chunk = df1_chunk[['matchId', 'participantId', 'win', 'gameDuration','role','lane','individualPosition']]
        df2_chunk=CreateVerticalDataframe(df2_chunk, 1)

        # Step 3b: Set the common key column as index (if not already)
        df1_chunk.set_index(['matchId', 'participantId'], inplace=True)
        df2_chunk.set_index(['matchId', 'participantId'], inplace=True)

        # Step 3c: Merge or join the DataFrames
        merged_chunk = pd.merge(df1_chunk, df2_chunk, left_index=True, right_index=True, how='inner', suffixes=('', '_df2'))
        # Alternatively, you can use join: merged_chunk = df1_chunk.join(df2_chunk, how='inner')

        # Step 3d: Append the merged chunk to the list
        dfs.append(merged_chunk)

    # Step 4: Concatenate all DataFrames in the list
    merged_df = pd.concat(dfs)

    return(merged_df)

#####################
def read_and_join_two_huge_dataframe2(file1, file2):

    df1 = pd.read_csv(file1, usecols=['matchId', 'participantId', 'win', 'gameDuration','role','lane','individualPosition','championName'])

    df2 = pd.read_csv(file2,)
    float64_cols = df2.select_dtypes(include='float64').columns

    # Convert float64 columns to float32
    df2[float64_cols] = df2[float64_cols].astype('float32')

    df1.set_index(['matchId', 'participantId'], inplace=True)

    df2.rename(columns={'match_id': 'matchId'},inplace=True)

    #df2_new=CreateVerticalDataframe(df2,  1)
    #df2_new.set_index(['matchId', 'participantId'], inplace=True)
    # Step 2: Initialize an empty list to store DataFrames
    dfs = []

    #merged_df = pd.merge(df1, df2_new, left_index=True, right_index=True, how='inner', suffixes=('', '_df2'))
    merged_df = pd.merge(df1, df2, how='inner', on = ['matchId', 'participantId'], left_on=None, right_on=None, suffixes=('', '_df2'))


    dfs.append(merged_df)


    # Step 3: Concatenate all DataFrames in the list
    merged_df = pd.concat(dfs)
    print(len(merged_df))

    return (merged_df)

def read_and_join_two_huge_dataframe_ranked(file1, file2,usedcolumn):

    df1 = pd.read_csv(file1)



    df2 = pd.read_csv(file2, usecols=usedcolumn)
    float64_cols = df2.select_dtypes(include='float64').columns

    # Convert float64 columns to float32
    df2[float64_cols] = df2[float64_cols].astype('float32')


    df2.rename(columns={'matchId': 'match_id'},inplace=True)

    df2.set_index(['match_id', 'participantId'], inplace=True)


    df1.set_index(['match_id', 'participantId'], inplace=True)



    #df2.set_index(['matchId', 'participantId'], inplace=True)
    # Step 2: Initialize an empty list to store DataFrames
    dfs = []

    #merged_df = pd.merge(df1, df2, on = ['match_id', 'participantId'], how='left', suffixes=('', '_df2'))
    merged_df = pd.merge(df1, df2, how='inner', on = ['match_id', 'participantId'], left_on=None, right_on=None, suffixes=('', '_df2'))


    dfs.append(merged_df)


    # Step 3: Concatenate all DataFrames in the list
    merged_df = pd.concat(dfs)
    print(len(merged_df))

    return (merged_df)

#######################
def find_RankedmatchID(file1,file2):
    df1 = pd.read_csv(file1,usecols=['matchId'])
    df2 = pd.read_csv(file2,usecols=['matchId'])
    merged_df = pd.merge(df1, df2, how='inner', on = ['matchId'], left_on=None, right_on=None, suffixes=('', '_df2'))
    merged_df=merged_df.drop_duplicates()
    return merged_df

def read_and_join_two_huge_dataframe3(file1, file2):

    df1 = pd.read_csv(file1)
    float64_cols = df1.select_dtypes(include='float64').columns

    # Convert float64 columns to float32
    df1[float64_cols] = df1[float64_cols].astype('float32')

    df2 = pd.read_csv(file2, usecols=['matchId', 'participantId', 'time_interval','role']
        )


    df1.set_index(['matchId', 'participantId'], inplace=True)
    df2.set_index(['matchId', 'participantId'], inplace=True)


    merged_df = pd.merge(df1, df2, how='inner', on = ['matchId', 'participantId','role'], left_on=None, right_on=None, suffixes=('', '_df2'))
    print(len(merged_df))
    ### make matchchallenge per phase that can be used for deep learning
    summary_df=make_matchchallenge_perphase(merged_df)

    return (summary_df)


def make_matchchallenge_perphase (df) :

    exlist=['participantId','participant_number','win']
    numeric_columns = [col for col in df.select_dtypes(include='number').columns if col not in exlist]
    agg_dict = {col: 'mean' for col in numeric_columns}

    # Grouping by 'mid' and 'pid', and aggregating the data
    summary_df = df.groupby(['matchId', 'time_interval', 'role','win']).agg(agg_dict).reset_index()
    return (summary_df)

def calculate_timePhase(output):
    df = pd.read_csv(output)
    print(len(df))
    # Sort the DataFrame by 'matchid' and 'Pid' to maintain the order
    df = df.sort_values(['match_id', 'participantId']).reset_index(drop=True)

    # Add a 'time' column that starts from 1 for each 'matchid' and 'Pid' combination
    df['time'] = df.groupby(['match_id', 'participantId']).cumcount() + 1

    # Add a 'phase' column based on the 'time' value
    df['Phase'] = pd.cut(
        df['time'],
        bins=[0, 10, 20, float('inf')],
        labels=[1, 2, 3],
        right=False
    )
    return (df)

###path for integrating all matchtimeline files into one master file
inputdata_path1 = "../../LOLDataset/MatchTimeline_Part3/*.csv"
outputdata_path1 = "../data/Output/MatchTimeline/MatchTimeline_Masterfiles/MatchTimeline_Masterfile_part3.csv"

##path for merging three matchtimeline_perPhaseMean to one file
inputdata_pathR1 = "../data/OutputRank/MatchTimeline/MatchTimeline_masterfile_positionRowdata1.csv"

inputdata_pathR2 = "../data/OutputRank/MatchTimeline/MatchTimeline_masterfile_positionRowdata2.csv"

outputdata_pathR=  "../data/OutputRank/MatchResume/MatchResume_Masterfile.csv"
outputdata_pathRFinal=  "../data/OutputRank/MatchResume/FinalTeamMatchResume_Masterfile.csv"
outputdata_pathPR=  "../data/OutputRank/MatchTimeline/MatchTimeline_masterfile_position.csv"
outputdata_pathMT = "../data/OutputRank/MatchTimeline/MatchTimeline_WithExtraColumns.csv"



inputdata_pathRanked = "../data/OutputRank/MatchResume/FinalRankedMatchResume_Masterfile.csv"
outputdata_pathRT="../data/OutputRank/MatchTimeline/RankedMatchTimeline_withExtraColumns.csv"

Inputdata_forMatchId = "../../RiotProject/data/Input/UniqueMatchId.csv"
outputdata_forMatchId="../../RiotProject/data/Input/UniqueMatchId_Ranked.csv"
inputdata_path2 = "../data/OutputRank/MatchTimeline/*.csv"
outputdata_path2 = "../data/OutputRank/MatchTimeline/MatchTimeline_Masterfile.csv"


##path for merging two MatchTimeline_Masterfiles to one file
inputdata_path3 = "../data/Output/MatchTimeline/MatchTimeline_Masterfiles/*.csv"
outputdata_path3 = "../data/Output/MatchTimeline/MatchTimeline_Masterfile.csv"

##path for merging two MatchChallenges_Masterfile to one file
inputdata_path4 = "../data/Output/MatchChallenges/MatchChallenges_Masterfiles/*.csv"
outputdata_path4 = "../data/Output/MatchChallenges/MatchChallenges_Masterfile.csv"

##path for merging  MATCHCHALLENGE with  matchtimeline masterfile
inputdata_path_file5 = "../data/Output/MatchChallenges/MatchChallenges_Masterfile.csv"
inputdata_path_file6 = "../data/Output/MatchTimeline/MatchTimeline_Masterfile.csv"

outputdata_path_file5 = "../data/Output/MasterFile/MatchTimeline_WithExtraColumns.csv"

##path for merging  matchtimeline_perPhaseMean with matchchallenge master file
inputdata_path_file7 = "../data/Output/MasterFile/MatchTimeline_PerMatchPhase.csv"

outputdata_path_file6 = "../data/Output/MasterFile/MatchChallenge_PerMatchPhase.csv"



outputdata_path_file8 = "../data/OutputRank/MatchTimeline/MatchTimeline_masterfile_withRole.csv"

inputdata_path_file9 = "../data/OutputRank/MatchTimeline/MatchTimeline_masterfile_position.csv"

outputdata_path_file9 = "../data/OutputRank/ClusterResults/ZoneperMinute_perParticipant.csv"

outputdata_path_filepositionphase = "../data/OutputRank/ClusterResults/ZoneperMinute_perParticipantPhase.csv"

outputdata_path_file10="../data/OutputRank/ClusterResults/ZoneperMinute_perRoleRank.csv"

outputdata_path_file11 = "../data/OutputRank/ClusterResults/ZoneperMinute_perRole.csv"

outputdata_path_file12 = "../data/OutputRank/MatchTimeline/RankedMatchTimeline_Masterfile.csv"

inputdata_pathMPR = "../data/OutputRank/MatchTimeline/MatchTimeline_masterfile_PositionwithRole.csv"

inputdata_positionRow="../data/OutputRank/MatchTimeline/MatchTimeline_masterfile_positionRowdata.csv"

outputdata_positionRowRole = "../data/OutputRank/MatchTimeline/MatchTimeline_masterfile_PositionwithRole2024.csv"

#######
outputdata_path2024Final=  "../data/OutputRank/MatchResume/FinalTeamMatchResume_Masterfile2024.csv"

outputdata_MatchTimeline2024Final=  "../data/OutputRank/MatchTimeline/MatchTimeline_masterfile_PositionwithRole2024.csv"

inputdata_MatchResume=  "../data/OutputRank/MatchResume/MatchResume_TeamMasterfile.csv"
outputdata_MatchResume2024Final=  "../data/OutputRank/MatchResume/MatchResume_TeamMasterfile2024.csv"


outputdata_RankedMatchResume2024Final=  "../data/OutputRank/MatchResume/RankedMatchResume_Masterfile2024.csv"

outputdata_RankedMatchTimeline2024Position=  "../data/OutputRank/MatchTimeline/MatchTimeline_masterfile_PositionwithRoleRank2024.csv"

outputdata_RankedMatchTimeline2024PositionRank="../data/OutputRank/MatchTimeline/MatchTimeline_masterfile_PositionRowwithRoleRankPhase2024New.csv"



df1=calculate_timePhase(outputdata_RankedMatchTimeline2024Position)
df1.to_csv(outputdata_RankedMatchTimeline2024PositionRank,index=True)


###Filter and get data of 2024
#filter_2024Data(outputdata_pathRFinal, outputdata_path2024Final)

#filter_2024TimekineData(outputdata_pathRFinal, outputdata_path2024Final)

#filter_2024MatchData(inputdata_MatchResume,outputdata_MatchResume2024Final)


#filter_2024MatchData(inputdata_pathMPR,outputdata_MatchTimeline2024Final)


#df.to_csv(inputdata_path_file9,index=False)

##df5=read_and_join_two_huge_dataframe_ranked(outputdata_pathR,outputdata_path2,['matchId', 'participantId', 'win', 'gameDuration','role','lane','individualPosition','championName'])
#df5.to_csv(outputdata_pathMT,index=True)

#df=pd.read_csv(outputdata_pathR)
#df['role'] = df['role'].replace('NONE', 'JUNGLE')
#df['lane'] = df['lane'].replace('NONE', 'JUNGLE')
#df.to_csv(outputdata_pathR)


#df1=read_and_merge_datafiles(inputdata_path1)
#df1.to_csv(outputdata_path1,index=False)


#df2=read_and_merge_dataframes(inputdata_path2)
#df2.to_csv(outputdata_path2,index=False)
#print (len(df2))
#df3=read_and_merge_dataframes(inputdata_path3)
#df3.to_csv(outputdata_path3,index=False)

#df4=read_and_merge_dataframes(inputdata_path4)
#df4.to_csv(outputdata_path4,index=False)


#df5=read_and_join_two_huge_dataframe2(inputdata_path_file5,inputdata_path_file6)
#df5.to_csv(outputdata_path_file5,index=True)

###not needed
#df6=read_and_join_two_huge_dataframe3(inputdata_path_file5,inputdata_path_file7)
#df6.to_csv(outputdata_path_file6,index=True)


#df8=read_and_join_two_huge_dataframe2(outputdata_pathR,inputdata_path_file8)
#df8.to_csv(outputdata_path_file8,index=True)

#df9=read_and_join_two_huge_dataframe2(outputdata_pathR,outputdata_path_file9)
#df9.to_csv(outputdata_path_file11,index=True)


#df6=read_and_join_two_huge_dataframe_ranked(inputdata_pathRanked,outputdata_path_file9,['matchId', 'participantId', 'adjustedPoints', 'rank','tier','role','lane','win','teamId'])
#df6.to_csv(outputdata_path_file10,index=True)

#df7=find_RankedmatchID(Inputdata_forMatchId,inputdata_pathRanked)
#df7.to_csv(outputdata_forMatchId,index=False)

#df6=read_and_join_two_huge_dataframe_ranked(inputdata_positionRow,outputdata_MatchResume2024Final,['matchId', 'participantId', 'role','lane','win','teamId','individualPosition','teamPosition'])
#df6.to_csv(outputdata_positionRowRole,index=True)

#df7=read_and_join_two_huge_dataframe_ranked(inputdata_positionRow,outputdata_RankedMatchResume2024Final,['matchId', 'participantId', 'adjustedPoints', 'rank','tier','role','lane','win','teamId','individualPosition','teamPosition'])
#df7.to_csv(outputdata_RankedMatchTimeline2024Position,index=True)