import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib.font_manager import FontProperties

def getname(phase):
    if phase==1:
        return 'Early game'
    if phase==2:
        return 'Mid game'
    if phase==3:
        return 'Late game'
    else:
        return phase
def ChampionKill_analysis(InputfilePath,outputPath):
    championevent_list=['CHAMPION_KILL_victimDamageDealt','CHAMPION_victimDamageRecived']
    # Get the top 10 names by frequency
    for event in championevent_list:
        Inputfilename = InputfilePath + event + '.csv'
        df = pd.read_csv(Inputfilename)

        top_names = df['name'].value_counts().nlargest(10).index

        # Filter the dataframe to include only the top 10 names
        df_top_names = df[df['name'].isin(top_names)]

        # Create a pivot table to count occurrences of each name in each phase
        pivot_table = df_top_names.pivot_table(index='name', columns='Phase', aggfunc='size', fill_value=0)

        # Add a column for the total count and sort the pivot table by this total
        pivot_table['Total'] = pivot_table.sum(axis=1)
        pivot_table = pivot_table.sort_values(by='Total', ascending=True).drop(columns='Total')

        outputfilename = outputPath + 'top 10' + event+'.csv'
        pivot_table.to_csv(outputfilename, index=True)


        # Plotting
        pivot_table.plot(kind='barh', stacked=True, figsize=(10, 7))
        plt.title('Top 10 champions' + event)
        plt.xlabel('Name')
        plt.ylabel('Count')
        plt.legend(title='Phase')
        plt.show()
# Create a figure and axis


def SequentionalPatternAnalysis(InputfilePath):
    Inputfilename = InputfilePath + 'CHAMPION_SPECIAL_KILL' + '.csv'
    df = pd.read_csv(Inputfilename)
    fig, ax = plt.subplots(figsize=(10, 6))

    df['multiKillLength'] = df['multiKillLength'].astype(str)
    df['killtyp'] = df['killType'] + '_' + df['multiKillLength']
    # Unique entities and events
    entities = df['killtyp'].unique()
    events = df['Phase'].unique()

    # Define a color palette
    palette = sns.color_palette("husl", len(events))

    # Create a color map for events
    event_color_map = {event: palette[i] for i, event in enumerate(events)}

    # Plot sequences for each entity
    for i, entity in enumerate(entities):
        entity_data = df[df['killtyp'] == entity]
        for j, row in entity_data.iterrows():
            ax.plot([row['timestamp'], row['timestamp']], [i - 0.4, i + 0.4],
                    color=event_color_map[row['Phase']], linewidth=10, solid_capstyle='butt')

    # Set the y-ticks to be the entity names
    ax.set_yticks(range(len(entities)))
    ax.set_yticklabels(entities)

    # Add a legend
    handles = [plt.Line2D([0, 1], [0, 0], color=color, linewidth=10, solid_capstyle='butt')
               for event, color in event_color_map.items()]
    labels = list(event_color_map.keys())
    ax.legend(handles, labels, title='Events')

    # Format the x-axis
    plt.xticks(rotation=45)
    plt.xlabel('Timestamp')
    plt.ylabel('KillType')
    plt.title('Event Sequences')

    plt.tight_layout()
    plt.show()


def MultiKill_Analysis(InputfilePath,outputPath):
    Inputfilename = InputfilePath + 'CHAMPION_SPECIAL_KILL' + '.csv'
    df = pd.read_csv(Inputfilename,usecols=['Phase','timestamp','multiKillLength','killType','position_x','position_y'])
    df_mkills=df[df['killType']=='KILL_MULTI']

    for phase in df_mkills['Phase'].unique():
        background_image = plt.imread('../RealLOLmap.jpg')
        df_mkills_phase = df_mkills[df_mkills['Phase'] == phase]
        for killlentype in df_mkills_phase['multiKillLength'].unique():
            df_mkills_pos = df_mkills_phase[df_mkills_phase['multiKillLength'] == killlentype]

            extent = 0, 15000, 0, 15000
            # Add the background image
            plt.imshow(background_image, extent=extent, aspect='auto')
            sns.scatterplot(x='position_x', y='position_y', s=1, data=df_mkills_pos, color='black', alpha=0.5)

            # Draw the density heatmap
            sns.kdeplot(x='position_x', y='position_y', data=df_mkills_pos, cmap='coolwarm', cut=0, fill=None, thresh=0,
                        levels=100,
                        alpha=0.5)
            # hb = plt.hexbin(df['position_x'], df['position_y'], gridsize=50, cmap='coolwarm', mincnt=1, alpha=0.6)
            # Add a color bar
            # cb = plt.colorbar(hb, label='Counts')

            plt.xlabel('X_position')
            plt.ylabel('Y_position')

            plt.title(f'density heatmap for {killlentype} in {getname(phase)} ')

            # Display the plot
            plt.show()

    ######################
    for phase in df_mkills['Phase'].unique():
        df_mkills_phase=df_mkills[df_mkills['Phase']==phase]
        df_mkills_phase['time_minutes'] = df_mkills_phase['timestamp'] / (1000 * 60)  # Convert milliseconds to minutes
        if phase==3 :
            bin=40
        else:
            bin=10
        bin_counts, bin_edges = np.histogram(df_mkills_phase['time_minutes'], bins=bin)
        bin_start_int = bin_edges[:-1].astype(int)
        # Create a DataFrame to save the bin edges and counts
        histogram_df = pd.DataFrame({
            'Bin': bin_start_int,  # Start of the bin
            'Count': bin_counts  # Count of values in the bin
        })
        grouped = df_mkills_phase.groupby('time_minutes')['multiKillLength'].mean().reset_index()
        # Store histogram data in a CSV file
        histogram_data = grouped[['time_minutes', 'multiKillLength']]
        outputfilename = outputPath + 'multiKillHistogram_phase'+ getname(phase) +'.csv'
        #histogram_data.to_csv(outputfilename, index=False)
        histogram_df.to_csv(outputfilename, index=False)
        # Plotting
        plt.figure(figsize=(10, 6))
        #plt.plot(grouped['time_minutes'], grouped['multiKillLength'], marker='o', linestyle='-')

        plt.hist(df_mkills_phase['time_minutes'], bins=10, edgecolor='black', alpha=0.7)

        # Customize plot
        plt.title(f'Distribution of Multi Kills Across Time Slots of phase {getname(phase)}')
        plt.xlabel('Time Slot (Minutes)')
        plt.ylabel('Number of Kills')


        plt.show()

    for killtype in df['killType'].unique():
        df_killtype= df[df['killType'] == killtype]
        for phase in df_killtype['Phase'].unique():
            background_image = plt.imread('../RealLOLmap.jpg')
            df_killtype_phase=df_killtype[df_killtype['Phase']==phase]
            extent = 0, 15000, 0, 15000
                # Add the background image
            plt.imshow(background_image, extent=extent, aspect='auto')
            sns.scatterplot(x='position_x', y='position_y', s=1, data=df_killtype_phase, color='black', alpha=0.5)

            # Draw the density heatmap
            sns.kdeplot(x='position_x', y='position_y', data=df_killtype_phase, cmap='coolwarm', cut=0,fill=None, thresh=0, levels=100,
                        alpha=0.5)
            # hb = plt.hexbin(df['position_x'], df['position_y'], gridsize=50, cmap='coolwarm', mincnt=1, alpha=0.6)
            # Add a color bar
            # cb = plt.colorbar(hb, label='Counts')

            plt.xlabel('X_position')
            plt.ylabel('Y_position')

            plt.title(f'density heatmap for {killtype} in {getname(phase)} ')

            # Display the plot
            plt.show()


def PositionBased_KillTypeEventAnalysis(InputfilePath):
    i = 0
    for event in event_Typelist:
        Inputfilename = InputfilePath + event + '.csv'
        df = pd.read_csv(Inputfilename)
        for cls in df['Phase'].unique():
            df_cluster = df[df['Phase'] == cls]
            #if len(df_cluster) > 1000000:
            #    df_cluster = df_cluster.sample(n=100000, random_state=1)  # Select 1000 random samples
            rank = event + 'for all'

            Visualize_LolMapBasedonPositionwithPercentage(df_cluster, cls, rank, event_TypeField[i])
        i = i + 1

def PositionBased_EventAnalysis(InputfilePath):
    for event in event_list:
        if event!='WARD_KILL':
            Inputfilename= InputfilePath+ event + '.csv'
            df=pd.read_csv(Inputfilename)
            for cls in df['Phase'].unique():
                df_cluster = df[df['Phase'] == cls]
                if len(df_cluster) > 100000:
                    df_cluster = df_cluster.sample(n=100000, random_state=1)  # Select 1000 random samples
                rank=event + 'for all'

                Visualize_LolMapBasedonPositionHeatmap(df_cluster, cls, rank)
               # Visualize_LolMapBasedonPositionwithPercentage(df_cluster, cls, rank,type)


def PositionBased_RankedEventAnalysis(InputfilePath):
    for event in event_list:
        if event!='WARD_KILL':
            Inputfilename= InputfilePath+ event + '.csv'
            df=pd.read_csv(Inputfilename)
            for cls in df['Phase'].unique():
                df_cluster = df[df['Phase'] == cls]
                for tier in df['tier'].unique():
                    df_tier = df_cluster[df_cluster['tier'] == tier]
                    if len(df_tier) > 100000:
                        df_tier = df_tier.sample(n=100000, random_state=1)  # Select 1000 random samples
                    rank=event +tier
                    Visualize_LolMapBasedonPositionHeatmap(df_tier, cls, rank)


def KillDistribution_Analysis(InputfilePath,outputPath):
    merged_df=pd.DataFrame()
    for event in event_list:
        Inputfilename = InputfilePath + event + '.csv'
        df = pd.read_csv(Inputfilename,usecols=['type','matchId','Phase'])
        merged_df = pd.concat([merged_df,df])

    grouped = merged_df.groupby(['type', 'Phase'])['matchId'].count().reset_index()
    outputfilename= outputPath + 'distribution of kills in game stages.csv'
    grouped.to_csv(outputfilename,index=False)

    # Plot using seaborn
    plt.figure(figsize=(10, 6))
    sns.barplot(data=grouped, x='Phase', y='matchId', hue='type')

    # Customize plot
    plt.title('Number of Kills in Different Phases')
    plt.xlabel('Phase')
    plt.ylabel('Count of Kills')
    plt.legend(title='Kill Type')

    # Show the plot
    plt.show()

def KillTemporal_MonsterKillPerRoleLane(InputfilePath,outputPath):
    event = 'ELITE_MONSTER_KILL'
    Inputfilename = InputfilePath + event + '.csv'
    df = pd.read_csv(Inputfilename)
    df['monstertyp'] = df['monsterType']+'_' + df['monsterSubType'].fillna('')

    ######################
    counts_role = df.groupby(['Phase', 'role']).size().reset_index(name='count')

    # Step 2: Calculate the total counts for each combination of P, L, and R
    total_counts = counts_role.groupby(['Phase'])['count'].transform('sum')

    # Step 3: Calculate the percentage
    counts_role['percentage'] = (counts_role['count'] / total_counts) * 100

    outputfile = outputPath + 'Total' + event + '_PerRoleNoType.csv'
    counts_role.to_csv(outputfile, index=True)
    ######################
    ######################
    counts_role = df.groupby(['Phase', 'lane']).size().reset_index(name='count')

    # Step 2: Calculate the total counts for each combination of P, L, and R
    total_counts = counts_role.groupby(['Phase'])['count'].transform('sum')

    # Step 3: Calculate the percentage
    counts_role['percentage'] = (counts_role['count'] / total_counts) * 100

    outputfile = outputPath + 'Total' + event + '_PerLaneNoType.csv'
    counts_role.to_csv(outputfile, index=True)
    ######################

    ######################
    counts_role = df.groupby(['Phase', 'role']).size().reset_index(name='count')

    # Step 2: Calculate the total counts for each combination of P, L, and R
    total_counts = counts_role.groupby(['Phase'])['count'].transform('sum')

    # Step 3: Calculate the percentage
    counts_role['percentage'] = (counts_role['count'] / total_counts) * 100

    outputfile = outputPath + 'Total' + event + '_PerRoleNoType.csv'
    counts_role.to_csv(outputfile, index=True)

    ######################
    counts_role = df.groupby(['Phase', 'monstertyp']).size().reset_index(name='count')

    # Step 2: Calculate the total counts for each combination of P, L, and R
    total_counts = counts_role.groupby(['Phase'])['count'].transform('sum')

    # Step 3: Calculate the percentage
    counts_role['percentage'] = (counts_role['count'] / total_counts) * 100

    outputfile = outputPath + 'Total' + event + '_PerType.csv'
    counts_role.to_csv(outputfile, index=True)
    ######################

def KillTemporal_SpecialKillPerRoleLane(InputfilePath, outputPath):
        event = 'CHAMPION_SPECIAL_KILL'
        Inputfilename = InputfilePath + event + '.csv'
        df = pd.read_csv(Inputfilename)

        df['multiKillLength'].fillna('0',inplace=True)
        df = df[df['multiKillLength'] == '0']
        df['multiKillLength']=df['multiKillLength'].astype(str)
        df['killtyp'] = df['killType'] + '_' + df['multiKillLength']

        ######################
        counts_role = df.groupby(['Phase', 'role']).size().reset_index(name='count')

        # Step 2: Calculate the total counts for each combination of P, L, and R
        total_counts = counts_role.groupby(['Phase'])['count'].transform('sum')

        # Step 3: Calculate the percentage
        counts_role['percentage'] = (counts_role['count'] / total_counts) * 100

        outputfile = outputPath + 'Total' + event + '_PerRoleNoType.csv'
        counts_role.to_csv(outputfile, index=True)
        ######################
        ######################
        counts_role = df.groupby(['Phase', 'lane']).size().reset_index(name='count')

        # Step 2: Calculate the total counts for each combination of P, L, and R
        total_counts = counts_role.groupby(['Phase'])['count'].transform('sum')

        # Step 3: Calculate the percentage
        counts_role['percentage'] = (counts_role['count'] / total_counts) * 100

        outputfile = outputPath + 'Total' + event + '_PerLaneNoType.csv'
        counts_role.to_csv(outputfile, index=True)
        ######################

        ######################
        counts_role = df.groupby(['Phase', 'role']).size().reset_index(name='count')

        # Step 2: Calculate the total counts for each combination of P, L, and R
        total_counts = counts_role.groupby(['Phase'])['count'].transform('sum')

        # Step 3: Calculate the percentage
        counts_role['percentage'] = (counts_role['count'] / total_counts) * 100

        outputfile = outputPath + 'Total' + event + '_PerRoleNoType.csv'
        counts_role.to_csv(outputfile, index=True)

        ######################
        counts_role = df.groupby(['Phase', 'killtyp']).size().reset_index(name='count')

        # Step 2: Calculate the total counts for each combination of P, L, and R
        total_counts = counts_role.groupby(['Phase'])['count'].transform('sum')

        # Step 3: Calculate the percentage
        counts_role['percentage'] = (counts_role['count'] / total_counts) * 100

        outputfile = outputPath + 'Total' + event + '_PerType.csv'
        counts_role.to_csv(outputfile, index=True)
        ######################

def KillTemporal_BuildingKillPerRoleLane(InputfilePath,outputPath):

    event='BUILDING_KILL'
    Inputfilename = InputfilePath + event + '.csv'
    df = pd.read_csv(Inputfilename)
    df['build_towerType'] = df['buildingType']+'_' + df['towerType'].fillna('')
    # Step 1: Calculate the count of each type for each combination of P, L, and R
    counts = df.groupby(['Phase', 'lane', 'role', 'build_towerType']).size().reset_index(name='count')

    # Step 2: Calculate the total counts for each combination of P, L, and R
    total_counts = counts.groupby(['Phase', 'lane', 'role'])['count'].transform('sum')

    # Step 3: Calculate the percentage
    counts['percentage'] = (counts['count'] / total_counts) * 100

    outputfile = outputPath + 'Total' + event + '_PerRoleLane.csv'
    counts.to_csv(outputfile, index=True)

    # Step 4: Create a bar plot to show the percentage distribution for each P
    g = sns.FacetGrid(counts, col='Phase', col_wrap=2, height=5, aspect=1.5)
    g.map_dataframe(sns.barplot, x='role', y='percentage', hue='build_towerType', ci=None, palette='viridis')

    # Customizing the plot
    g.set_axis_labels('role', 'Percentage')
    g.set_titles('Phase = {col_name}')
    g.add_legend(title='building type')
    plt.show()

    ######################
    counts_lane = df.groupby(['Phase', 'lane',  'build_towerType']).size().reset_index(name='count')

    # Step 2: Calculate the total counts for each combination of P, L, and R
    total_counts = counts_lane.groupby(['Phase', 'lane'])['count'].transform('sum')

    # Step 3: Calculate the percentage
    counts_lane['percentage'] = (counts_lane['count'] / total_counts) * 100

    outputfile = outputPath + 'Total' + event + '_PerLane.csv'
    counts_lane.to_csv(outputfile, index=True)
    ######################

    # Step 4: Create a bar plot to show the percentage distribution for each P
    g = sns.FacetGrid(counts_lane, col='Phase', col_wrap=2, height=5, aspect=1.5)
    g.map_dataframe(sns.barplot, x='lane', y='percentage', hue='build_towerType', ci=None, palette='viridis')

    # Customizing the plot
    g.set_axis_labels('role', 'Percentage')
    g.set_titles('Phase = {col_name}')
    g.add_legend(title='building type')
    plt.show()
#####################################
  ######################
    counts_role = df.groupby(['Phase', 'role',  'build_towerType']).size().reset_index(name='count')

    # Step 2: Calculate the total counts for each combination of P, L, and R
    total_counts = counts_role.groupby(['Phase', 'role'])['count'].transform('sum')

    # Step 3: Calculate the percentage
    counts_role['percentage'] = (counts_role['count'] / total_counts) * 100

    outputfile = outputPath + 'Total' + event + '_PerRole.csv'
    counts_role.to_csv(outputfile, index=True)


    ######################
    counts_role = df.groupby(['Phase', 'role']).size().reset_index(name='count')

    # Step 2: Calculate the total counts for each combination of P, L, and R
    total_counts = counts_role.groupby(['Phase'])['count'].transform('sum')

    # Step 3: Calculate the percentage
    counts_role['percentage'] = (counts_role['count'] / total_counts) * 100

    outputfile = outputPath + 'Total' + event + '_PerRoleNoType.csv'
    counts_role.to_csv(outputfile, index=True)
    ######################
    ######################
    counts_role = df.groupby(['Phase', 'lane']).size().reset_index(name='count')

    # Step 2: Calculate the total counts for each combination of P, L, and R
    total_counts = counts_role.groupby(['Phase'])['count'].transform('sum')

    # Step 3: Calculate the percentage
    counts_role['percentage'] = (counts_role['count'] / total_counts) * 100

    outputfile = outputPath + 'Total' + event + '_PerLaneNoType.csv'
    counts_role.to_csv(outputfile, index=True)
    ######################

    ######################
    counts_role = df.groupby(['Phase', 'role']).size().reset_index(name='count')

    # Step 2: Calculate the total counts for each combination of P, L, and R
    total_counts = counts_role.groupby(['Phase'])['count'].transform('sum')

    # Step 3: Calculate the percentage
    counts_role['percentage'] = (counts_role['count'] / total_counts) * 100

    outputfile = outputPath + 'Total' + event + '_PerRoleNoType.csv'
    counts_role.to_csv(outputfile, index=True)

    ######################
    counts_role = df.groupby(['Phase', 'build_towerType']).size().reset_index(name='count')

    # Step 2: Calculate the total counts for each combination of P, L, and R
    total_counts = counts_role.groupby(['Phase'])['count'].transform('sum')

    # Step 3: Calculate the percentage
    counts_role['percentage'] = (counts_role['count'] / total_counts) * 100

    outputfile = outputPath + 'Total' + event + '_PerType.csv'
    counts_role.to_csv(outputfile, index=True)
    ######################

def FindMostFrequentWardEvents(filename):
    ####################
    df = pd.read_csv(filename)
    #df=df[df['Phase']==1]

    # Convert time from milliseconds to minutes
    df['time_min'] = df['timestamp'] / (1000 * 60)  # Convert milliseconds to minutes

    # Define bins based on the phase
    df['time_bin'] = None  # Initialize an empty column for time bins

    # Apply different binning rules based on the phase
    # For Phase 1: Time between 0 to 9 minutes
    df.loc[df['Phase'] == 1, 'time_bin'] = pd.cut(df[df['Phase'] == 1]['time_min'], bins=range(0, 10), right=False)

    # For Phase 2: Time between 10 to 19 minutes
    df.loc[df['Phase'] == 2, 'time_bin'] = pd.cut(df[df['Phase'] == 2]['time_min'], bins=range(10, 20), right=False)

    # For Phase 3: Time greater than 20 minutes
    df.loc[df['Phase'] == 3, 'time_bin'] = pd.cut(df[df['Phase'] == 3]['time_min'],
                                                  bins=range(20, int(df['time_min'].max()) + 2), right=False)

    # Count occurrences for each type within each time bin and phase
    time_bin_counts = df.groupby(['Phase', 'type', 'time_bin']).size().reset_index(name='count')

    # Generate all possible combinations of phases, types, and time bins
    all_bins = pd.DataFrame([(phase, t, b) for phase in df['Phase'].unique()
                             for t in df['type'].unique()
                             for b in pd.cut(range(0, 10), bins=range(0, 10), right=False) if phase == 1
                             or b in pd.cut(range(10, 20), bins=range(10, 20), right=False) if phase == 2
                             or b in pd.cut(range(20, int(df['time_min'].max()) + 2),
                                            bins=range(20, int(df['time_min'].max()) + 2), right=False) if phase == 3],
                            columns=['Phase', 'type', 'time_bin'])

    # Group by phase, type, time_bin, and mat to get counts
    grouped_counts = df.groupby(['Phase', 'type', 'time_bin', 'matchId']).size().reset_index(name='count')

    # Calculate the average count across all 'mat' for each combination of phase, type, and time_bin
    average_counts = grouped_counts.groupby(['Phase', 'type', 'time_bin'])['count'].mean().reset_index(
        name='average_count')


    outputfile = outputdata_path + 'MostFrequentWardEvents_Phase1.csv'
    average_counts.to_csv(outputfile, index=True)

def StatisticalAnalysis_Ward(filename1,filename2,filename3):

    event='ALLWARD'
    df = pd.read_csv(filename1)


    df_group_permatchId = df.groupby(['Phase', 'matchId']).size().reset_index(name='count')
    stats_per_phase = df_group_permatchId.groupby('Phase')['count'].agg(['mean', 'std', 'count']).reset_index()

    stats_per_phase['sem'] = stats_per_phase['std'] / np.sqrt(stats_per_phase['count'])

    stats_per_phase.rename(
        columns={'mean': 'average_records_per_id', 'std': 'std_records_per_id', 'count': 'num_ids'}, inplace=True)

    outputfile = outputdata_path + 'Total' + event + '_PerPhase.csv'
    stats_per_phase.to_csv(outputfile, index=True)

    # Group by 'phase' and 'type' to get the counts
    grouped = df.groupby(['Phase', 'type']).size().reset_index(name='count')

    # Calculate the total count for each phase
    total_counts = df.groupby('Phase').size().reset_index(name='total_count')

    # Merge the counts with the total counts
    merged = pd.merge(grouped, total_counts, on='Phase')

    # Calculate the percentage
    merged['percentage'] = (merged['count'] / merged['total_count']) * 100

    outputfile = outputdata_path + 'Total' + event + '_PerTypePhase.csv'
    merged.to_csv(outputfile, index=True)





    ###Rankedevents######################
    df = pd.read_csv(filename2)
    # Group by 'phase' and 'type' to get the counts
    grouped = df.groupby(['Phase', 'tier']).size().reset_index(name='count')

    # Calculate the total count for each phase
    total_counts = df.groupby('Phase').size().reset_index(name='total_count')

    # Merge the counts with the total counts
    merged = pd.merge(grouped, total_counts, on='Phase')

    # Calculate the percentage
    merged['percentage'] = (merged['count'] / merged['total_count']) * 100

    outputfile = outputdata_path + 'Total' + event + '_PerRankPhase.csv'
    merged.to_csv(outputfile, index=True)

    ###Rankedevents######################
    df_rank = pd.read_csv(filename3)

    # Calculate total count for each rank to normalize
    rank_totals = df_rank.groupby('tier').size().reset_index(name='total_rank_count')

    # Merge the total rank counts with the original dataframe
    df = pd.merge(df, rank_totals, on='tier')

    # Count occurrences for each combination of phase and rank
    phase_rank_counts = df.groupby(['Phase', 'tier']).size().reset_index(name='phase_rank_count')

    # Merge to add total count for each rank
    phase_rank_counts = pd.merge(phase_rank_counts, rank_totals, on='tier')

    # Calculate normalized percentage for each rank within each phase
    phase_rank_counts['normalized_percentage'] = (phase_rank_counts['phase_rank_count'] / phase_rank_counts[
        'total_rank_count']) * 100

    print(phase_rank_counts)

    outputfile = outputdata_path + 'Total' + event + '_PerRankPhase.csv'
    phase_rank_counts.to_csv(outputfile, index=True)



def KillTemporal_GamePhaseAnalysis(InputfilePath,outputPath):
    for event in event_list:
        Inputfilename= InputfilePath+ event + '.csv'
        df=pd.read_csv(Inputfilename)

        df_group_permatchId=df.groupby(['Phase','matchId']).size().reset_index(name='count')
        stats_per_phase=df_group_permatchId.groupby('Phase')['count'].agg(['mean','std','count']).reset_index()

        stats_per_phase['sem']=stats_per_phase['std']/np.sqrt(stats_per_phase['count'])

        stats_per_phase.rename(
        columns={'mean': 'average_records_per_id', 'std': 'std_records_per_id', 'count': 'num_ids'}, inplace=True)



        outputfile=outputPath+'Total'+ event+'_PerPhase.csv'
        stats_per_phase.to_csv(outputfile,index=True)

        stats_per_phase.plot(kind='bar')
        plt.xlabel('Game Phase')
        plt.ylabel(f'Number of {event}')
        plt.title(f'Kills Distribution of {event} by Game Phase')
        plt.show()

def KillTemporal_GameRankPhaseAnalysis_championspecialKill(InputfilePath,outputPath):
        Inputfilename= InputfilePath+ 'CHAMPION_SPECIAL_KILL' + '.csv'
        df=pd.read_csv(Inputfilename)
        df=df[df['killType']=='KILL_FIRST_BLOOD']
        plt.figure(figsize=(14, 8))
        ranks_order = ["IRON", "BRONZE", "SILVER", "GOLD", "PLATINUM", "EMERALD","DIAMOND", "MASTER", "GRANDMASTER", "CHALLENGER"]

        df_group_permatchId = df.groupby(['Phase', 'matchId','tier']).size().reset_index(name='count')
        stats_per_phase = df_group_permatchId.groupby(['Phase','tier'])['count'].agg(['mean', 'std', 'count']).reset_index()

        stats_per_phase['sem'] = stats_per_phase['std'] / np.sqrt(stats_per_phase['count'])

        outputfile=outputPath+'Total'+ 'CHAMPION_SPECIAL_KILL'+'_KILL_FIRST_BLOODPerPhaseRank.csv'
        stats_per_phase.to_csv(outputfile,index=True)

        # Step 3: Create a bar plot
        fig, ax = plt.subplots(figsize=(10, 6))

        sns.barplot(data=stats_per_phase, x='tier', y='count', hue='Phase', palette=sns,order=ranks_order)
        # Loop through each value of 'p' and plot the average counts for different values of 'r'

        # Customizing the plot
        ax.set_xlabel('tier')
        ax.set_ylabel('Average Records per ID')
        ax.set_title('Average Number of Records per ID for Different Values of tier and phase')
        ax.legend(title='tier')
        plt.show()



def Visualize_LolMapBasedonPositionHeatmap(df,type,rank):
    background_image = plt.imread('../RealLOLmap.jpg')
    x = df['position_x']
    y = df['position_y']

    #extent = np.min(x), np.max(x), np.min(y), np.max(y)
    extent=0,15000,0,15000
    # Add the background image
    plt.imshow(background_image, extent=extent, aspect='auto')
    sns.scatterplot(x='position_x', y='position_y', s=1, data=df, color='black', alpha=0.5)
    #plt.scatter(df['position_x'], df['position_y'], s=1, color='black', alpha=0.3)
    # Draw the density heatmap
    sns.kdeplot(x='position_x', y='position_y', data=df, cmap='coolwarm', fill=None, thresh=0, levels=100, alpha=0.5)
    #hb = plt.hexbin(df['position_x'], df['position_y'], gridsize=50, cmap='coolwarm', mincnt=1, alpha=0.6)
    # Add a color bar
    #cb = plt.colorbar(hb, label='Counts')

    plt.xlabel('X_position')
    plt.ylabel('Y_position')

    plt.title(f'density heatmap for {rank} in {getname(type)} ')

        # Display the plot
    plt.show()



def Visualize_LolMapBasedonPositionwithPercentage(df,phase,rank,type):
    background_image = plt.imread('../Real_LOLmap.png')
    x = df['position_x']
    y = df['position_y']

    # Calculate the percentage of each Type
    type_counts = df[type].value_counts()
    total_count = len(df)
    type_percentages = (type_counts / total_count) * 100
    plt.figure(figsize=(10, 8))
    #extent = np.min(x), np.max(x), np.min(y), np.max(y)
    extent=-600,15000,0,15000
    # Add the background image
    plt.imshow(background_image, extent=extent, aspect='auto')

    scatter=sns.scatterplot(x='position_x', y='position_y', data=df, hue=type, palette='tab10')#color='black', alpha=0.5)

    # Create custom legend labels with percentages
    handles, labels = scatter.get_legend_handles_labels()
    #custom_labels = [f'{label} ({type_percentages[label]:.2f}%)' for label in
    #                 labels[0:]]
    #percent_labels = [f'{type_percentages[label]:.2f}%' for label in labels[0:]]
    # Add the legend with custom labels
    font_prop = FontProperties(weight='bold')
    #plt.legend(handles=handles[0:], labels=custom_labels, title=type, bbox_to_anchor=(1.00, 1), loc='upper right')
    #legend = plt.gca().get_legend()

    #for text, pct_label in zip(legend.get_texts(), percent_labels):
    #    text.set_fontweight('bold')
    #    text.set_text(f'{text.get_text()} ({pct_label})')
    #    text.set_color('black')  # Change the color of the percentage text

    plt.xlabel('X_position')
    plt.ylabel('Y_position')

    plt.title(f'Scatter Plot for {rank} in {getname(phase)} ')


        # Display the plot
    plt.show()

def JoinForChampionforwardEvents(file1, file2,usedcolumn):

    df1 = pd.read_csv(file1)
    df1['creatorId']=df1['creatorId'].fillna(0)
    df1_creator=df1[df1['creatorId']!=0]

    df1_creator['creatorId'] = df1_creator['creatorId'].astype(int)



    df2 = pd.read_csv(file2, usecols=usedcolumn)
    float64_cols = df2.select_dtypes(include='float64').columns

    # Convert float64 columns to float32
    df2[float64_cols] = df2[float64_cols].astype('float32')




    df2.set_index(['matchId', 'participantId'], inplace=True)


    dfs=[]

    merged_df = pd.merge(df1_creator, df2, how='inner', left_on=['matchId', 'creatorId'], right_on=['matchId', 'participantId'])
    dfs.append(merged_df)

    df1['killerId']=df1['killerId'].fillna(0)
    df1_killer=df1[df1['killerId']!=0]

    df1_killer['killerId'] = df1_killer['killerId'].astype(int)

    merged_df = pd.merge(df1_killer, df2, how='inner', left_on=['matchId', 'killerId'], right_on=['matchId', 'participantId'])

    dfs.append(merged_df)
    # Step 3: Concatenate all DataFrames in the list
    merged_df = pd.concat(dfs)
    print(len(merged_df))

    return (merged_df)

event_list=['ELITE_MONSTER_KILL','WARD_KILL','CHAMPION_SPECIAL_KILL','BUILDING_KILL','CHAMPION_KILL']

event_Typelist=['CHAMPION_SPECIAL_KILL','BUILDING_KILL','ELITE_MONSTER_KILL']

event_TypeField=['multiKillLength','towerType','monsterType']

inputMatchEvent_path = "../data/OutputRank/MatchEvents/MatchEvent_MasterFile_"

MatchEvent_Wardpath = "../data/OutputRank/MatchEvents/MatchEvent_MasterFile_WARD_Events.csv"

MatchEvent_RankedWardpath = "../data/OutputRank/MatchEvents/MatchEvent_MasterFile_RANKED_ALLWARDs.csv"
inputMatchEvent_Rankedpath = "../data/OutputRank/MatchEvents/MatchEvent_MasterFileRanked_"
#outputMatchEvent_path = "../data/OutputRank/MatchEvents/MatchEvent_MasterFile_CHAMPION_KILL_main_Phase.csv"
outputdata_path= "../data/OutputRank/ReadyToPlot/EventAnalysis_"

inputpath_rankedmasterfile="../data/OutputRank/MatchResume/FinalRankedMatchResume_Masterfile.csv"

inputpath_teamResumeMasterfile="../data/OutputRank/MatchResume/MatchResume_TeamMasterfile2024.csv"

OutputMatchEvent_path = "../data/OutputRank/MatchEvents/"
###Total Kill events per different game phases
#KillTemporal_GamePhaseAnalysis(inputMatchEvent_path,outputdata_path)

#KillTemporal_GameRankPhaseAnalysis(inputMatchEvent_Rankedpath,outputdata_path)
#KillTemporal_GameRankPhaseAnalysis_championspecialKill(inputMatchEvent_Rankedpath,outputdata_path)
#ChampionKill_analysis(inputMatchEvent_path,outputdata_path)

#PositionBased_EventAnalysis(inputMatchEvent_path)

#PositionBased_RankedEventAnalysis(inputMatchEvent_Rankedpath)

#PositionBased_KillTypeEventAnalysis(inputMatchEvent_path)

#KillDistribution_Analysis(inputMatchEvent_path,outputdata_path)

#MultiKill_Analysis(inputMatchEvent_path,outputdata_path)

##KillTemporal_BuildingKillPerRoleLane(inputMatchEvent_Rankedpath,outputdata_path)

#KillTemporal_MonsterKillPerRoleLane(inputMatchEvent_Rankedpath,outputdata_path)

#KillTemporal_SpecialKillPerRoleLane(inputMatchEvent_Rankedpath,outputdata_path)


#Sequentinal Data Analysis
#SequentionalPatternAnalysis(inputMatchEvent_Rankedpath)

#StatisticalAnalysis_Ward(MatchEvent_Wardpath,MatchEvent_RankedWardpath)

#FindMostFrequentWardEvents(MatchEvent_Wardpath)

df=JoinForChampionforwardEvents(MatchEvent_Wardpath,inputpath_teamResumeMasterfile,['matchId', 'participantId', 'win', 'gameDuration','role','lane','individualPosition','championName'])
df.to_csv(OutputMatchEvent_path+"WardEventsWithChampions.csv",index=False)
