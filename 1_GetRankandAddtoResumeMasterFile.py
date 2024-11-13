import pandas as pd

import shutil
import os
import glob
from datetime import datetime
import time


def GetplayerRank_fromrankfiles(file1, file2):

    df1 = pd.read_csv(file1) ## file including puuid and gamename and tag
    df1['summonerName'] = df1['gameName']+'#'+df1['tagLine']

    df2 = pd.read_csv(file2)
    #float64_cols = df2.select_dtypes(include='float64').columns

    # Convert float64 columns to float32
    #df2[float64_cols] = df2[float64_cols].astype('float32')

    df1.set_index([ 'puuid'], inplace=True)

    df2.set_index(['puuid'], inplace=True)
    # Step 2: Initialize an empty list to store DataFrames


    merged_df = pd.merge(df1, df2, how='inner', on = ['puuid'], left_on=None, right_on=None, suffixes=('', '_df2'))




    # Step 3: Concatenate all DataFrames in the list

    print(len(merged_df))

    return (merged_df)
#######################

def create_matchResumewithPlayerRank(file1, file2):
    df1 = pd.read_csv(file1)

    df2 = pd.read_csv(file2)

    # df1.set_index([ 'puuid'], inplace=True)

    # df2.set_index(['puuid'], inplace=True)

    df2['summonerName'] = df2['summonerName'].str.split('#').str[0]

    # summorname=df1['summonerName'].iloc[0]
    # print (puid)
    # df_new=df2[df2['summonerName'] ==puid]

    # merged_df = pd.merge(df1, df2_new, left_index=True, right_index=True, how='inner', suffixes=('', '_df2'))
    merged_df = pd.merge(df1, df2, on='summonerName', how='left')

    # pd.merge(df1, df2, on='key', how='left')
    print(len(merged_df))

    return (merged_df)


#######################

def create_matchResumewithPlayerRank(file1, file2):

    df1 = pd.read_csv(file1)

    df2 = pd.read_csv(file2,encoding='latin-1')

    gamename=df2['summonerName'].unique()
    print (len(gamename))

    #df1.set_index([ 'puuid'], inplace=True)

    #df2.set_index(['puuid'], inplace=True)

    #df2['summonerName']=df2['summonerName'].str.split('#').str[0]


    #summorname=df1['summonerName'].iloc[0]
   # print (puid)
   # df_new=df2[df2['summonerName'] ==puid]
   

    #merged_df = pd.merge(df1, df2, left_index=True, right_index=True, how='inner', suffixes=('', '_df2'))
    merged_df = pd.merge(df1, df2, on = 'summonerName', how='left')

   # pd.merge(df1, df2, on='key', how='left')
    print(len(merged_df))

    return (merged_df)
#######################

###path for integrating all matchtimeline files into one master file
inputdata_path1 = "../../RiotProject/models/MatchId.csv"
inputdata_path2 = "../../RiotProject/models/SummonerIdFromNames.csv"



inputdata_path3 = "../data/OutputRank/MatchResume/MatchResume_TeamMasterfile2024.csv"





outputdata_path1 = "../../RiotProject/models/MatchUserRank.csv"

outputdata_path2 = "../../data/OutputRank/MatchResume/MatchResume_MasterfilewithRank2.csv"

outputdata_path3 = "../data/OutputRank/MatchResume/RankedMatchResume_Masterfile2024.csv"


##files of Giano fro getting rank of players
outputdata_path5 = "../../RiotProject/data/ranks/ranksOfthePlayerspart3.csv"

outputdata_path6 = "../../RiotProject/models/SummonerId_whitoudRanks_information_Original.csv"

outputdata_path7 = "../../RiotProject/models/SummonerWithRanksAdjustedPoints.csv"

#df=create_matchIDwithPlayerRank(inputdata_path1, inputdata_path2)
#df.to_csv(outputdata_path1,index=True)

"""
df_rank=GetplayerRank_fromrankfiles(outputdata_path6, outputdata_path5)  #file with summonername and rank
df_rank.to_csv(outputdata_path7,index=True)

"""
#
df_total=create_matchResumewithPlayerRank(inputdata_path3,outputdata_path7)
#df2=create_matchResumewithPlayerRank(inputdata_path4,outputdata_path7)
#df2.to_csv(outputdata_path2,index=True)
#df_total=pd.concat([df1,df2],axis=0,ignore_index=False)

#df_total.to_csv(outputdata_path2,index=True)
print (len(df_total))
df_total=df_total[df_total['rank'].notnull()]
print (len(df_total))
df_total.to_csv(outputdata_path3,index=True)
