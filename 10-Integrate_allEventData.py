
### detail analysis of evens that occures during the game
import pandas as pd
import os
import glob

import re

def categorize_phase(timestamp):
    if timestamp < earlygame_tr:
        return 1
    elif timestamp < midgame_tr:
        return 2
    else:
        return 3
def remove_digit_prefix(df):

    for col in df.columns:
        if col!='timestamp':
            new_col_name = '_'.join(col.split('_')[2:])  # Remove the digit_ pattern
            df.rename(columns={col: new_col_name}, inplace=True)

    return (df)


def getRankofPlayerforEventdata (InputfilePath,InputRankedfile,outputfilepath):
    df_rank = pd.read_csv(InputRankedfile, usecols=['matchId','participantId','rank','tier','role','individualPosition','lane'])
    df_rank.rename(columns={'participantId': 'killerId'}, inplace=True)

    df_rank.set_index(['matchId', 'killerId'], inplace=True)
    for event in event_list:
        ### read event file
        Inputfilename = InputfilePath + event + '.csv'
        df_event = pd.read_csv(Inputfilename)

        merged_df = pd.merge(df_event, df_rank, how='inner', on = ['matchId', 'killerId'], left_on=None, right_on=None, suffixes=('', '_df2'))
        output_filename=outputfilepath + event + '.csv'
        merged_df.to_csv(output_filename,index=False)
        print (len(merged_df),event)

def getRankofPlayerforWardEventdata (Inputfilename,InputRankedfile,outputfilepath):
    ##WARD_KILL
    df_rank = pd.read_csv(InputRankedfile, usecols=['matchId','participantId','rank','tier','role','individualPosition','lane'])
    df_event = pd.read_csv(Inputfilename)

    df1=df_event[df_event['type']=='WARD_KILL']
    df_rank.rename(columns={'participantId': 'killerId'}, inplace=True)

    df_rank.set_index(['matchId', 'killerId'], inplace=True)


    df_merged1 = pd.merge(df1, df_rank, how='inner', on=['matchId', 'killerId'], left_on=None, right_on=None,
                         suffixes=('', '_df2'))


    ##WARD_PLACED
    df_rank = pd.read_csv(InputRankedfile, usecols=['matchId','participantId','rank','tier','role','individualPosition','lane'])


    df2=df_event[df_event['type']=='WARD_PLACED']


    df_rank.rename(columns={'participantId': 'creatorId'}, inplace=True)

    df_rank.set_index(['matchId', 'creatorId'], inplace=True)



    df_merged2 = pd.merge(df2, df_rank, how='inner', on=['matchId', 'creatorId'], left_on=None, right_on=None,
                         suffixes=('', '_df2'))

    # Combine the results to get the final DataFrame
    df_final = pd.concat([df_merged1,df_merged2], axis=0, ignore_index=True)

    #df_final.rename(columns={'killerId': 'participantId'}, inplace=True)
    outputfilename=outputfilepath+'RANKED_ALLWARDs.csv'
    df_final.to_csv(outputfilename)

def get_championKillData(df):
    # Create the first dataframe with columns that have only names
    columns_list=['victimDamageDealt','victimDamageReceived']
    df_main_cols= [col for col in df.columns if not any(sub in col for sub in columns_list)]
    digit_no=[1,2,3,4,5,6,7,8,9,10,11,12]
    # Create the first dataframe
    df_main = df[df_main_cols].copy()
    df_main['Phase'] = df_main['timestamp'].apply(categorize_phase)

    #df_main['idx'] = df_main.index
    matchId=df_main.iloc[0]['matchId']
    # Function to create a dataframe for AA and BB columns
    def create_group_df(df, prefix):
        #group_cols = [col for col in df.columns if col.startswith(prefix)]
       # df_group=df[group_cols]
        #group_df = pd.DataFrame()
        participant_data_list = []
        for digit_number in digit_no:
            # Select columns with the current participant number
            str1=prefix+'_'+ str(digit_number)
            selected_columns_notime = [column for column in df.columns if column.startswith(f'{str1}_') ]
            selected_columns=selected_columns_notime
            selected_columns.append('timestamp')
            # Create a new DataFrame with selected columns
            participant_data = df[selected_columns]

            # Append the participant's data to the list
            # Move the last column(s) to the position of the first column(s)
            participant_data = pd.concat([participant_data.iloc[:, -1:], participant_data.iloc[:, :-1]], axis=1)

            # Remove rows where all columns except 'timestamp' are empty


            participant_data = remove_digit_prefix(participant_data)
            participant_data['p_idx'] = digit_number
            participant_data['matchId']=matchId
            participant_data['Phase'] = participant_data['timestamp'].apply(categorize_phase)

            participant_data_list.append(participant_data)

        participant_data = pd.concat(participant_data_list, axis=0, ignore_index=True)
        non_time_cols = [col for col in participant_data.columns if col not in ['timestamp','matchId','p_idx','Phase']]
        participant_data = participant_data.dropna(how='all', subset=non_time_cols)
        return participant_data

    # Create dataframe for AA columns
    df_aa_group = create_group_df(df, columns_list[0])

    # Create dataframe for BB columns
    df_bb_group = create_group_df(df, columns_list[1])
    return df_main,df_aa_group, df_bb_group


def integrate_allEventFiles(InputfilePath,OutputfilePath):
    csv_files = []
    csv_files = glob.glob(InputfilePath)


    files_limit =71000
    files_processed = 0

    dfs_champion1 = []
    dfs_champion2 = []
    dfs_champion3 = []
    dfs_building=[]
    dfs_championSpecial=[]
    dfs_eliteMonster=[]
    dfs_ward=[]
    # Iterate through each CSV file and append its data to the combined DataFrame
    for csv_file in csv_files:
        # Read the CSV file into a DataFrame
        df = pd.read_csv(csv_file)
        for event in event_list:

            filtered_df = df[df['type']==event]
            if len(filtered_df) > 0:
                if event =='CHAMPION_KILL':
                    filtered_df = filtered_df.dropna(axis=1, how='all')

                    df_champion1,df_champion2,df_champion3=get_championKillData(filtered_df)
                    dfs_champion1.append(df_champion1)
                    dfs_champion2.append(df_champion2)
                    dfs_champion3.append(df_champion3)

                if event=='BUILDING_KILL':
                    filtered_df = filtered_df.dropna(axis=1, how='all')
                    filtered_df['Phase'] = filtered_df['timestamp'].apply(categorize_phase)
                    dfs_building.append(filtered_df)

                if event=='CHAMPION_SPECIAL_KILL':
                    filtered_df = filtered_df.dropna(axis=1, how='all')
                    filtered_df['Phase'] = filtered_df['timestamp'].apply(categorize_phase)
                    dfs_championSpecial.append(filtered_df)

                if event=='ELITE_MONSTER_KILL':
                    filtered_df = filtered_df.dropna(axis=1, how='all')
                    filtered_df['Phase'] = filtered_df['timestamp'].apply(categorize_phase)
                    dfs_eliteMonster.append(filtered_df)

                if event=='WARD_KILL' or event=='WARD_PLACED':
                    filtered_df = filtered_df.dropna(axis=1, how='all')
                    filtered_df['Phase'] = filtered_df['timestamp'].apply(categorize_phase)
                    dfs_ward.append(filtered_df)

        files_processed += 1
        print(files_processed)
        if files_processed >= files_limit:
           break
           #print(files_processed)
        else:
           print(csv_file)

    if len(dfs_ward):
        combined_df = pd.concat(dfs_ward, ignore_index=True)
        Outputfilename=OutputfilePath+'WARD_KILL.csv'
        combined_df.to_csv(Outputfilename, index=False)

    if len(dfs_eliteMonster):
        combined_df = pd.concat(dfs_eliteMonster, ignore_index=True)
        Outputfilename=OutputfilePath+'ELITE_MONSTER_KILL.csv'
        combined_df.to_csv(Outputfilename, index=False)

    if len(dfs_championSpecial):
        combined_df = pd.concat(dfs_championSpecial, ignore_index=True)
        Outputfilename=OutputfilePath+'CHAMPION_SPECIAL_KILL.csv'
        combined_df.to_csv(Outputfilename, index=False)

    # Concatenate all DataFrames in the list into a single DataFrame
    if len(dfs_champion1):
        combined_df = pd.concat(dfs_champion1, ignore_index=True)
        Outputfilename=OutputfilePath+'CHAMPION_KILL.csv'
        combined_df.to_csv(Outputfilename, index=False)

    if len(dfs_champion2):
        combined_df = pd.concat(dfs_champion2, ignore_index=True)
        Outputfilename=OutputfilePath+'CHAMPION_KILL_victimDamageDealt.csv'
        combined_df.to_csv(Outputfilename, index=False)

    if len(dfs_champion3):
        combined_df = pd.concat(dfs_champion3, ignore_index=True)
        Outputfilename=OutputfilePath+'CHAMPION_KILL_victimDamageRecived.csv'
        combined_df.to_csv(Outputfilename, index=False)

    if len(dfs_building):
        combined_df = pd.concat(dfs_building, ignore_index=True)
        Outputfilename=OutputfilePath+'BUILDING_KILL.csv'
        combined_df.to_csv(Outputfilename, index=False)



def integrate_allWARDEventdatafromFiles(InputfilePath,OutputfilePath):
    csv_files = []
    csv_files = glob.glob(InputfilePath)


    files_limit =200000
    files_processed = 0

    dfs_ward=[]
    # Iterate through each CSV file and append its data to the combined DataFrame
    for csv_file in csv_files:
        # Read the CSV file into a DataFrame
        df = pd.read_csv(csv_file)
        filtered_df=df[df['type'].isin(['WARD_KILL','WARD_PLACED'])]
        filtered_df['Phase'] = filtered_df['timestamp'].apply(categorize_phase)
        filtered_df = filtered_df.dropna(axis=1, how='all')

        dfs_ward.append(filtered_df)

        files_processed += 1
        print(files_processed)
        if files_processed >= files_limit:
           break
           #print(files_processed)
        else:
           print(csv_file)

    if len(dfs_ward):
        combined_df = pd.concat(dfs_ward, ignore_index=True)
        Outputfilename=OutputfilePath+'WARD_Events.csv'
        combined_df.to_csv(Outputfilename, index=False)

###############################
###INTEGRATE ALL ITEM RELATED EVENTS
def integrate_allEventdatafromFiles(InputfilePath,OutputfilePath,eventfield,feventlist):
    csv_files = []
    csv_files = glob.glob(InputfilePath)


    files_limit =200000
    files_processed = 0

    dfs_event=[]
    # Iterate through each CSV file and append its data to the combined DataFrame
    for csv_file in csv_files:
        # Read the CSV file into a DataFrame
        df = pd.read_csv(csv_file)
        filtered_df=df[df['type'].isin(feventlist)]
        filtered_df['Phase'] = filtered_df['timestamp'].apply(categorize_phase)
        filtered_df = filtered_df.dropna(axis=1, how='all')

        dfs_event.append(filtered_df)

        files_processed += 1
        print(files_processed)
        if files_processed >= files_limit:
           break
           #print(files_processed)
        else:
           print(csv_file)

    if len(dfs_event):
        combined_df = pd.concat(dfs_event, ignore_index=True)
        Outputfilename=OutputfilePath+eventfield+'.csv'
        combined_df.to_csv(Outputfilename, index=False)

inputMatchEvent_path = "../../RiotProject/data/OutputMatchEvents/*.csv"
outputMatchEvent_path = "../data/OutputRank/MatchEvents/MatchEvent_MasterFile_"

outputMatchEventranked_path = "../data/OutputRank/MatchEvents/MatchEvent_MasterFileRanked_"

inputpath_EventMasterfile = "../data/OutputRank/MatchEvents/MatchEvent_MasterFile_"
inputpath_rankedmasterfile="../data/OutputRank/MatchResume/FinalRankedMatchResume_Masterfile.csv"

Inputfilename="../data/OutputRank/MatchEvents/MatchEvent_MasterFile_WARD_Events.csv"


earlygame_tr=600000
midgame_tr=1200000

item_eventlist=['ITEM_PURCHASED','ITEM_DESTROYED','ITEM_SOLD','ITEM_UNDO']

event_list=['WARD_KILL','ELITE_MONSTER_KILL','CHAMPION_SPECIAL_KILL','BUILDING_KILL','CHAMPION_KILL']
##integrating all eventfiles for specific events
#integrate_allEventFiles(inputMatchEvent_path,outputMatchEvent_path)


integrate_allEventdatafromFiles(inputMatchEvent_path,outputMatchEvent_path,'Events_Item',item_eventlist)
#integrate_allWARDEventdatafromFiles(inputMatchEvent_path,outputMatchEvent_path)
#getRankofPlayerforEventdata(inputpath_EventMasterfile,inputpath_rankedmasterfile,outputMatchEventranked_path)


#getRankofPlayerforWardEventdata (Inputfilename,inputpath_rankedmasterfile,outputMatchEvent_path)