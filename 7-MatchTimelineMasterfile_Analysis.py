import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

###For Non Ranked dataset
def make_matchtimeline_phaseSummary (df) :
    # Binning the time values into intervals
    bins = [0, 9, 20, float('inf')]
    labels = ['EarlyPhase', 'MidPhase', 'LatePhase']
    df['time_interval'] = pd.cut(df['time'], bins=bins, labels=labels, right=False)
    exlist=['participantId','participant_number','win']
    numeric_columns = [col for col in df.select_dtypes(include='number').columns if col not in exlist]
    agg_dict = {col: 'mean' for col in numeric_columns}
    agg_dict['role'] = 'first'  # Add the first role aggregation
    agg_dict['lane'] = 'first'
    agg_dict['individualPosition'] = 'first'
    agg_dict['win'] = 'first'
    agg_dict['championName'] = 'first'

    # Grouping by 'mid' and 'pid', and aggregating the data
    summary_df = df.groupby(['matchId', 'participantId', 'time_interval']).agg(agg_dict).reset_index()

    df = summary_df.dropna(subset=['win'])
    print(len(df),'after removing null')
    df['win'] = df['win'].astype(int)
    df['role'] = df['role'].replace('NONE', 'JUNGLE')
    df['lane'] = df['lane'].replace('NONE', 'JUNGLE')
    return (df)

###FOR Ranked dataset
def make_matchtimeline_phaseSummaryRanked (df) :



    df['win'] = df['win'].astype(int)

    df1 = df[df['participantId'] <6]  ###partcipant 1-5
    df2 = df[df['participantId'] >5]  ###partcipant 6-10


    exlist=['participantId','Phase']
    numeric_columns = [col for col in df.select_dtypes(include='number').columns if col not in exlist]
    agg_dict = {col: 'mean' for col in numeric_columns}
    #agg_dict['role'] = 'first'  # Add the first role aggregation
    #agg_dict['lane'] = 'first'
    #agg_dict['individualPosition'] = 'first'
    #agg_dict['win'] = 'first'
    #agg_dict['championName'] = 'first'

    # Grouping by 'mid' and 'pid', and aggregating the data
    summary_df1 = df1.groupby(['matchId', 'Phase']).agg(agg_dict).reset_index()
    summary_df2 = df2.groupby(['matchId', 'Phase']).agg(agg_dict).reset_index()

    summary_df=pd.concat([summary_df1,summary_df2],axis=0)

    return (summary_df)


def make_matchtimeline_phaseRankedPlayers (df) :



    df['win'] = df['win'].astype(int)

    df1 = df[df['participantId'] <6]  ###partcipant 1-5
    df2 = df[df['participantId'] >5]  ###partcipant 6-10


    exlist=['participantId','Phase']
    numeric_columns = [col for col in df.select_dtypes(include='number').columns if col not in exlist]
    agg_dict = {col: 'mean' for col in numeric_columns}
    #agg_dict['role'] = 'first'  # Add the first role aggregation
    #agg_dict['lane'] = 'first'
    #agg_dict['individualPosition'] = 'first'
    #agg_dict['win'] = 'first'
    #agg_dict['championName'] = 'first'

    # Grouping by 'mid' and 'pid', and aggregating the data
    summary_df1 = df1.groupby(['matchId','participantId', 'Phase']).agg(agg_dict).reset_index()
    summary_df2 = df2.groupby(['matchId', 'Phase']).agg(agg_dict).reset_index()

    summary_df=pd.concat([summary_df1,summary_df2],axis=0)

    return (summary_df)

###FOR Ranked dataset Role Summary Ranked
###create matchtimeline masterfile including two rows per match then joining with matchresume masterfile

def make_matchtimeline_RoleSummaryRanked (df) :
    df['role'] = df['role'].replace('NONE', 'JUNGLE')
    df['lane'] = df['lane'].replace('NONE', 'JUNGLE')

    df['team_id'] = df['participantId'].apply(lambda x: 100 if x < 5 else 200)

    print ('total number of rows for timeline per role/phase', len(df))
    exlist = ['participantId',  'win','matchId','Phase','Unnamed: 0','team_id']

    numeric_columns = [col for col in df.select_dtypes(include='number').columns if col not in exlist]
    cols_for_Last = [col for col in numeric_columns if col not in cols_for_Avg]

    agg_dict = {col: 'mean' for col in cols_for_Avg} | {col: 'last' for col in cols_for_Last}
    agg_dict['win'] = 'first'


    # Grouping by 'mid' and 'pid', and aggregating the data
    summary_df = df.groupby(['matchId', 'team_id','role']).agg(agg_dict).reset_index()

    df = summary_df.dropna(subset=['win'])
    print(len(df), 'after removing null')
    df['win'] = df['win'].astype(int)


    return (df)
def make_matchtimeline_ParticipantSummaryRanked (df) :

    exlist = ['participantId',  'win','matchId','Phase','Unnamed: 0']

    numeric_columns = [col for col in df.select_dtypes(include='number').columns if col not in exlist]
    cols_for_Last = [col for col in numeric_columns if col not in cols_for_Avg]

    agg_dict = {col: 'mean' for col in cols_for_Avg} | {col: 'last' for col in cols_for_Last}
    agg_dict['role'] = 'first'  # Add the first role aggregation
    agg_dict['lane'] = 'first'
    agg_dict['individualPosition'] = 'first'
    agg_dict['win'] = 'first'
    agg_dict['championName'] = 'first'

    # Grouping by 'mid' and 'pid', and aggregating the data
    summary_df = df.groupby(['matchId', 'participantId']).agg(agg_dict).reset_index()

    df = summary_df.dropna(subset=['win'])
    print(len(df), 'after removing null')
    df['win'] = df['win'].astype(int)
    df['role'] = df['role'].replace('NONE', 'JUNGLE')
    df['lane'] = df['lane'].replace('NONE', 'JUNGLE')

    return (df)

def make_matchtimeline_phaseStats (df) :

    df['role'] = df['role'].replace('NONE', 'JUNGLE')
    df['lane'] = df['lane'].replace('NONE', 'JUNGLE')

    exlist=['participantId','participant_number','win']
    numeric_columns = [col for col in df.select_dtypes(include='number').columns if col not in exlist]
    agg_dict = {col: 'mean' for col in numeric_columns}
    #agg_dict['role']='size'
    summary_df1 = df.groupby(['time_interval', 'role', 'lane','win']).agg(agg_dict).reset_index()

    summary_df2 = df.groupby(['time_interval', 'role', 'lane', 'win']).agg({'matchId':'size','minionsKilled':'mean','totalGold':'mean','level':'mean','goldPerSecond':'mean'}).reset_index()
    make_plots(summary_df2)
    return (summary_df1,summary_df2)

def make_plots(df):
    sns.barplot(data=df, x='role', y='totalGold', hue='time_interval')
    plt.title('Comparison of Column totalGold for Different Time Intervals ')
    plt.xlabel('role')
    plt.ylabel('totalGold')
    plt.show()

    sns.barplot(data=df, x='role', y='goldPerSecond', hue='time_interval')
    plt.title('Comparison of Column goldPerSecond for Different Time Intervals ')
    plt.xlabel('role')
    plt.ylabel('goldPerSecond')
    plt.show()


    sns.barplot(data=df[df['win']==True], x='role', y='goldPerSecond', hue='time_interval')
    plt.title('Comparison of Column goldPerSecond for Different Time Intervals -Winner ')
    plt.xlabel('role')
    plt.ylabel('goldPerSecond')
    plt.show()

    sns.barplot(data=df[df['win']==False], x='role', y='goldPerSecond', hue='time_interval')
    plt.title('Comparison of Column goldPerSecond for Different Time Intervals -losser ')
    plt.xlabel('role')
    plt.ylabel('goldPerSecond')
    plt.show()
##path for merging two matchtimeline_perPhaseMean to one file
inputdata_path1 = "data/OutputRank/MatchTimeline/MatchTimeline_WithExtraColumns.csv"
outputdata_path1 = "data/OutputRank/MatchTimeline/MatchTimeline_PerMatchPhase.csv"
outputdata_path2 = "data/OutputRank/MatchTimeline/MatchTimeline_PerMatchRole.csv"
outputdata_path3 = "data/OutputRank/MatchTimeline/MatchTimeline_PerMatchParticipant.csv"


inputdata_path2 = "data/Output/MasterFile/MatchTimeline_PerMatchPhase.csv"
outpdata_path2 = "data/Output/MasterFile/MatchTimeline_MatchPhaseStats1.csv"
outpdata_path3 = "data/Output/MasterFile/MatchTimeline_MatchPhaseStats2.csv"

inputdata_pathforSelectedCols = "data/Input/MatchTimeline_FinalCols.csv"
df_cols = pd.read_csv(inputdata_pathforSelectedCols)

# Convert the DataFrame column to a Python array
cols_name = df_cols.iloc[:, 0].tolist()


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


##1----making a summary of matchtimeline per phase and role
df = pd.read_csv(inputdata_path1)
##
df_phase=make_matchtimeline_phaseSummaryRanked(df)
#print(df.groupby(['matchId']).size())
#df_phase.to_csv(outputdata_path1, index=True)

df_role=make_matchtimeline_RoleSummaryRanked(df)
df_role.to_csv(outputdata_path2, index=True)

#df_role=make_matchtimeline_ParticipantSummaryRanked(df)
#df_role.to_csv(outputdata_path3, index=True)

#df_phase=make_matchtimeline_phaseSummaryPerParticipant(df)
#print(df.groupby(['matchId']).size())
#df_phase.to_csv(outpdata_path2, index=True)

##2----getting stats for matchtimeline per phase
#df = pd.read_csv(inputdata_path2)

#df_phase1,df_phase2=make_matchtimeline_phaseStats(df)
#df_phase1.to_csv(outpdata_path2, index=True)
#df_phase2.to_csv(outpdata_path3, index=True)
