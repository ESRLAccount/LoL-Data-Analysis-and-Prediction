## this file include Exploratory Data Analysis (EDA) for some attribute in match resume and math time line , finding
## vion plots ( relation ship between some variables with target variable


import pandas as pd
import numpy as np

from scipy import stats
import seaborn as sns
import matplotlib.pyplot as plt

def create_countPlot(df,f1,f2,description):
    custom_palette = {
        'YELLOW_TRINKET': 'yellow',
        'BLUE_TRINKET': 'blue',
        'SIGHT_WARD': 'green',
        'CONTROL_WARD': 'red',
        'UNDEFINED': 'purple'
    }
    # 1. Countplot: Showing counts of wardtypes across game phases
    plt.figure(figsize=(10, 6))
    sns.countplot(data=df, x=f1, hue=f2, palette=custom_palette)
    plt.title(f'{f2} Across{f1}')
    plt.show()

    # 2. Stacked Bar Plot
    ward_phase_counts = df.groupby([f1, f2]).size().unstack()
    ward_phase_counts.plot(kind='bar', stacked=True, color=[custom_palette[col] for col in ward_phase_counts.columns], figsize=(10, 6))
    plt.title(f'Stacked {f2} in Each {f1}')
    plt.ylabel('Count')
    plt.show()

    # 3. Heatmap
    heatmap_data = df.groupby([f1, f2]).size().unstack().fillna(0)
    plt.figure(figsize=(10, 6))
    sns.heatmap(heatmap_data, annot=True, cmap="YlGnBu")
    plt.title(f'Heatmap of {f2} in Each {f1}')
    plt.show()

    # 4. Crosstab
    crosstab_result = pd.crosstab(df[f1], df[f2])
    output_filename=outputdata_path + f1 + '-' + f2 + '_'+ description+ '.csv'
    crosstab_result.to_csv(output_filename, index=True)
def create_vioinPlots (data,f1,f2):
    # Create a sample dataset similar to the description

    # Calculate mean and standard deviation
    """
    group_stats = data.groupby(f1)[f2].agg(['mean', 'std']).reset_index()

    # Merge the group stats back to the original dataframe
    data = data.merge(group_stats, on=f1)

    # Filter out values greater than 2 standard deviations for each group
    data = data[(data[f2] >= data['mean'] - 2.5 * data['std']) & (data[f2] <= data['mean'] + 2.5 * data['std'])]

    # Drop the extra 'mean' and 'std' columns after filtering if not needed
    data = data.drop(['mean', 'std'], axis=1)




    result_data = data[[f1,f2]]
    output_filename=outputdata_path + f1 + '-' + f2 + '.csv'
    result_data.to_csv(output_filename, index=True)
    """
    palette = {
        'TOP': '#1f77b4',  # Blue
        'UNDEFINED': '#9467bd',  # Purple
        'SIGHT_WARD': '#2ca02c',  # Green
        'CONTROL_WARD': '#d62728',  # Red
        'YELLOW_TRINKET': '#ffcc00'  # yellow
    }

    # Create the violin plot using seaborn
    """
    g = sns.FacetGrid(data, col='Phase', height=6, aspect=1)
    g.map(sns.violinplot, 'win', 'movement', palette='muted')

    # Set the axis labels and title for each plot
    g.set_axis_labels('Win', 'movement')
    g.set_titles('Phase: {col_name}')
    g.fig.suptitle('Violin Plots of Baronassist vs Win, Separated by Phase', y=1.02)

    # Show the plot
    plt.show()
    """



    plt.figure(figsize=(14, 8))
    #sns.violinplot(x='time_bin', y='movement', data=data,
     #              palette='muted', split=True)

    sns.violinplot(x=f1, y=f2, data=data, palette='muted')

    ax = plt.gca()

    # Add ticker borders to each violin
    for violin in ax.patches:  # ax.patches contains all the elements of the plot
        violin.set_edgecolor('black')  # Set border color
        violin.set_linewidth(2)  # Set border thickness

    # Set plot labels and title
    plt.title(f'{f2} Distribution Over different game {f1}')
    plt.xlabel(f1)
    plt.ylabel(f2)

    # Rotate x-axis labels for better readability
    plt.xticks(rotation=45)

    plt.show()


def create_vioinPlots_Phase (data,f1,f2):
    # Create a sample dataset similar to the description

    palette = {
        'TOP': '#1f77b4',  # Blue
        'UNDEFINED': '#9467bd',  # Purple
        'SIGHT_WARD': '#2ca02c',  # Green
        'CONTROL_WARD': '#d62728',  # Red
        'YELLOW_TRINKET': '#ffcc00'  # yellow
    }

    plt.figure(figsize=(10, 6))
    sns.violinplot(data=data, x='Phase', y=f2, hue=f1, split=True)

    # Add vertical lines to separate phases
    phases = data['Phase'].unique()
    for i in range(1, len(phases)):
        plt.axvline(i - 0.5, color='gray', linestyle='--', linewidth=1)

    # Set the axis labels and title for each plot
    plt.title('Distribution of Gold by Phase for Win and Loss')
    plt.xlabel('Game Phase')
    plt.ylabel(f2)

    # Show the plot
    plt.show()

def create_barplotWards(f1,data,description):
    count_df = data[f1].value_counts().reset_index()
    count_df.columns = [f1, 'count']
    plt.figure(figsize=(14, 8))
    sns.barplot(x=f1, y='count', data=count_df)

    output_filename=outputdata_path + 'count of each wardtype_'+description+ '.csv'
    count_df.to_csv(output_filename, index=False)

    plt.show()
def create_vioinPlots_Ward (data,f1,f2,palette):
    # Create a sample dataset similar to the description
    np.random.seed(42)  # For reproducibility

    result_data = data[[f1,f2]]
    output_filename=outputdata_path + f1 + '-' + f2 + '.csv'
    #result_data.to_csv(output_filename, index=False)



    plt.figure(figsize=(10, 6))

    sns.violinplot(x=f1, y=f2, data=data, palette=palette, density_norm="count")
    #sns.boxplot(x=f1, y=f2, data=data, palette='muted')  # palette)

        # Access the current axis
    ax = plt.gca()

    # Add ticker borders to each violin
    for violin in ax.patches:  # ax.patches contains all the elements of the plot
        violin.set_edgecolor('black')  # Set border color
        violin.set_linewidth(1.5)  # Set border thickness

    # Set plot labels and title
    plt.title(f'{f2} Distribution Over different game {f1}')
    plt.xlabel(f1)
    plt.ylabel(f2)

    # Rotate x-axis labels for better readability
    plt.xticks(rotation=45)

    plt.show()

def create_countPlot_Item(df,f1,f2,description):
    custom_palette = {
        '3340.0': 'yellow',
        '3363.0': 'blue',
        '2044.0': 'green',
        '2055.0': 'red',
        '3364.0': 'purple',
        '4643.0':'pink',
        'None':'black'

    }

    # 1. Countplot: Showing counts of wardtypes across game phases
    plt.figure(figsize=(10, 6))
    sns.countplot(data=df, x=f1, hue=f2, palette=custom_palette)
    plt.title(f'{f2} Across{f1} for {description}')
    plt.show()

    # 2. Stacked Bar Plot
    ward_phase_counts = df.groupby([f1, f2]).size().unstack()
    ward_phase_counts.plot(kind='bar', stacked=True, color=[custom_palette[col] for col in ward_phase_counts.columns], figsize=(10, 6))
    plt.title(f'Stacked {f2} in Each {f1} for  {description}')
    plt.ylabel('Count')
    plt.show()

    # 3. Heatmap
    heatmap_data = df.groupby([f1, f2]).size().unstack().fillna(0)
    plt.figure(figsize=(10, 6))
    sns.heatmap(heatmap_data, annot=True, cmap="YlGnBu")
    plt.title(f'Heatmap of {f2} in Each {f1} for {description}')
    #plt.show()

    # 4. Crosstab
    crosstab_result = pd.crosstab(df[f1], df[f2])
    output_filename=outputdata_path + f1 + '-' + f2 + '.csv'
    crosstab_result.to_csv(output_filename, index=False)

    print(crosstab_result)


inputMatchEvent_path1 = "../data/OutputRank/Spatio-temporal Analysis/"


inputMatchEvent_path2 = "../data/OutputRank/MatchResume/"
inputMatchEvent_path3 = "../data/OutputRank/MatchTimeline/"
inputMatchEvent_path4 = "../data/OutputRank/MatchEvents/"

outputdata_path= "../data/OutputRank/ReadyToPlot/EDA"
vision_columnLists=['visionScore','enemyVisionPings','wardsPlaced']
item_valueList=['2044.0','3340.0','2055.0','3363.0','3364.0','4643.0']



"""
filename=inputMatchEvent_path1 + "MovementPerParticipant.csv"
df=pd.read_csv(filename)

create_vioinPlots(df,'lane','movement')

####total distance between players
filename=inputMatchEvent_path1 + "total distance between players in each teamRowdata.csv"
df=pd.read_csv(filename)
create_vioinPlots(df,'lane','totalmovement distance per phase')

### composit vision score

filename=inputMatchEvent_path2 + "RankedMatchResume_Masterfile2024.csv"
df=pd.read_csv(filename)
df['composite vision score'] = df[vision_columnLists].mean(axis=1)

create_vioinPlots(df,'individualPosition','composite vision score')

#create_vioinPlots(df,'win','composite vision score')

filename=inputMatchEvent_path4 + "MatchEvent_MasterFile_WARD_Events.csv"
df=pd.read_csv(filename)

df.loc[df['wardType'] =='TEEMO_MUSHROOM' , 'wardType'] = 'UNDEFINED'

palette = {
    'BLUE_TRINKET': '#1f77b4',  # Blue
    'UNDEFINED': '#9467bd',  # Purple
    'SIGHT_WARD': '#2ca02c',  # Green
    'CONTROL_WARD': '#d62728',  # Red
    'YELLOW_TRINKET': '#ffcc00'   # yellow
}



filename=inputMatchEvent_path4 + "WardEventsWithChampions.csv"
df=pd.read_csv(filename)


palette = {
    'BLUE_TRINKET': '#1f77b4',  # Blue
    'UNDEFINED': '#9467bd',  # Purple
    'SIGHT_WARD': '#2ca02c',  # Green
    'CONTROL_WARD': '#d62728',  # Red
    'YELLOW_TRINKET': '#ffcc00'   # yellow
}
df.loc[df['wardType'] =='TEEMO_MUSHROOM' , 'wardType'] = 'UNDEFINED'

create_barplotWards('wardType',df,'All')

create_countPlot(df,'Phase','wardType','All')

df1=df[df['creatorId']>0]
create_barplotWards('wardType',df1,'Ward Placement')
create_countPlot(df1,'individualPosition','wardType','Ward Placement')
df2=df[df['killerId']>0]
create_barplotWards('wardType',df2,'Ward Destruction')
create_countPlot(df2,'individualPosition','wardType','Ward Destruction')

create_vioinPlots_Ward(df,'Phase','wardType',palette)
df=df[df['killerId']>0]
create_countPlot(df,'Phase','wardType','Ward Destruction')

df=df[df['creatorId']>0]
create_countPlot(f,'Phase','wardType','Ward Placement')

create_vioinPlots_Ward(df,'individualPosition','wardType',palette)

#df_grouped=df.groupby(['wardType','championName']).agg({'championName':'count'})
#print(df_grouped)
#group_A_X = df_grouped.get_group(('TEEMO_MUSHROOM', 'UNDEFINED'))
#print (group_A_X)
#df_grouped.to_csv(inputMatchEvent_path4 + 'creatorId.csv',index=True)
#for group, data in df_grouped:
#    print(f"Group: {group}")
    #print(data)

#####################Item Analysis for ward placed and destroyed
filename=inputMatchEvent_path4 + "MatchEvent_MasterFile_Events_Item.csv"
df=pd.read_csv(filename)
df['itemId']=df['itemId'].astype(str)
df_grouped=df.groupby(['type','itemId']).agg({'itemId':'count'})
print(df_grouped)
df_grouped.to_csv(inputMatchEvent_path4 + 'ItemsPerTypeCount.csv',index=True)
filter_df=df[df['itemId'].isin(item_valueList)]

filter_df1=filter_df[filter_df['type']=='ITEM_PURCHASED']
create_countPlot_Item (filter_df1,'Phase','itemId','ITEM_PURCHASED')

filter_df2=filter_df[filter_df['type']=='ITEM_DESTROYED']
create_countPlot_Item (filter_df2,'Phase','itemId','ITEM_DESTROYED')



filter_df3=filter_df[filter_df['type']=='ITEM_SOLD']
create_countPlot_Item (filter_df3,'Phase','itemId','ITEM_SOLD')

#####################EDA for KPI per match outcome##########
##most important metrics for match outcome

filename=inputMatchEvent_path2 + "MatchResume_TeamMasterfile2024.csv"
df=pd.read_csv(filename)
KPI_lists=['assists', 'baronKills','turretKills','turretsLost','physicalDamageTaken','bountyLevel','physicalDamageDealt',
 'totalDamageDealt', 'totalDamageDealtToChampions','totalDamageTaken', 'totalEnemyJungleMinionsKilled', 'totalHeal',
           'totalHealsOnTeammates', 'totalMinionsKilled', 'totalTimeCCDealt', 'totalTimeSpentDead', 'totalUnitsHealed',
           'tripleKills', 'trueDamageDealt', 'trueDamageDealtToChampions', 'trueDamageTaken', 'turretKills',
           'turretTakedowns', 'turretsLost', 'unrealKills', 'visionClearedPings', 'visionScore',
           'wardsKilled', 'wardsPlaced' ,'damageDealtToObjectives', 'damageDealtToTurrets', 'dangerPings',
           'deaths', 'detectorWardsPlaced', 'doubleKills', 'dragonKills', 'eligibleForProgression', 'enemyMissingPings',
           'enemyVisionPings',  'getBackPings', 'goldEarned', 'goldSpent', 'holdPings', 'inhibitorKills',
           'inhibitorTakedowns', 'inhibitorsLost' ,'kills', 'magicDamageDealt', 'magicDamageDealtToChampions',
           'magicDamageTaken', 'needVisionPings', 'neutralMinionsKilled', 'nexusKills', 'nexusLost',
           'nexusTakedowns', 'objectivesStolen', 'objectivesStolenAssists', 'onMyWayPings']

for KPI in KPI_lists:
    create_vioinPlots(df,'win',KPI)
"""
#####################EDA for KPI per match outcome##########
##most important metrics for match outcome

filename=inputMatchEvent_path3 + "MatchTimeline_PerMatchPhase.csv"
filename1=inputMatchEvent_path3 + "MatchTimeline_PerMatchPhase1.csv"
df=pd.read_csv(filename)
KPI_lists=['damageStats_trueDamageTaken','totalGold','championStats_power','xp','level','currentGold','goldPerSecond','totalGold','championStats_powerRegen']

for KPI in KPI_lists:
    create_vioinPlots_Phase(df,'win',KPI)
