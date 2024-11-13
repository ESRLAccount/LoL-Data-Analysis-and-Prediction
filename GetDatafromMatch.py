import pandas as pd

def CreateTeamInfo(df,dataset_name,matchId):
    Team_numbers = set(int(column.split('_')[0]) for column in df.columns if column[0].isdigit())

    # Extract and concatenate each participant's data
    team_data_list = []

    Team_numbers=['1','2']

    for team_number in Team_numbers:
        # Select columns with the current participant number
        selected_columns = [column for column in df.columns if column.split('_')[1].startswith(f'{team_number}_')]

        # Create a new DataFrame with selected columns
        team_data = df[selected_columns]


        # Remove the participant number from column names
        team_data.columns = [column.split('_')[2] for column in selected_columns]

        team_data['team_number'] = team_number
        # Append the participant's data to the list
        # Move the last column(s) to the position of the first column(s)
        team_data = pd.concat([team_data.iloc[:, -1:], team_data.iloc[:, :-1]], axis=1)

        team_data['match_id']= matchId


        team_data_list.append(team_data)

    # Concatenate the participant data in a row-wise manner
    concatenated_data = pd.concat(team_data_list, axis=0, ignore_index=True)


    concatenated_data.to_csv(output_path+dataset_name+'.csv',index=False)

def CreateVerticalDataframe(df,dataset_name,index,matchId):
    participant_numbers = set(int(column.split('_')[0]) for column in df.columns if column[0].isdigit())

    # Extract and concatenate each participant's data
    participant_data_list = []


    for participant_number in participant_numbers:
        # Select columns with the current participant number
        selected_columns = [column for column in df.columns if column.startswith(f'{participant_number}_')]

        # Create a new DataFrame with selected columns
        participant_data = df[selected_columns]


        # Remove the participant number from column names
        participant_data.columns = [column.split('_')[index] for column in selected_columns]

        participant_data['participant_number'] = participant_number
        # Append the participant's data to the list
        # Move the last column(s) to the position of the first column(s)
        participant_data = pd.concat([participant_data.iloc[:, -1:], participant_data.iloc[:, :-1]], axis=1)

        participant_data['match_id']= matchId


        participant_data_list.append(participant_data)

    # Concatenate the participant data in a row-wise manner
    concatenated_data = pd.concat(participant_data_list, axis=0, ignore_index=True)


    concatenated_data.to_csv(output_path+dataset_name+'.csv',index=False)
###main
input_path='data/Input/'
output_path='data/Output/'


#### matchTimeline
# Load the CSV file into a DataFrame
df = pd.read_csv(input_path+'MatchTimeline.csv')
matchId=df.iloc[0,-1]
df.drop('matchId',axis=1,inplace=True)
df=df.iloc[:,1:]


#Rootcolumns = [column.split('_')[1] for column in df.columns ]
#subset_df = df[Rootcolumns]
CreateVerticalDataframe(df, 'MatchTimeline_PerParticipant', 1,matchId)


#### matchResume
# Load the CSV file into a DataFrame

df = pd.read_csv(input_path+'MatchResume.csv')

# Array of dataset names
matchTimeline_datasetnames = ['team']  # Update with your actual dataset names

# Select columns without an underscore
Rootcolumns = [col for col in df.columns if col.split('_')[0] not in matchTimeline_datasetnames]

subset_df = df[Rootcolumns]


subset_df.to_csv(output_path+'matchinfo' + '.csv', index=False)


# Extract and store each dataset in a separate CSV file
for dataset_name in matchTimeline_datasetnames:
    # Select columns with the current dataset name
    selected_columns = [col for col in df.columns if col.split('_')[0] == dataset_name]


    # Create a new DataFrame with selected columns
    subset_df = df[selected_columns]

    subset_df['match_id'] = matchId

    CreateTeamInfo(subset_df, 'TeamStats',  matchId)



# Load the CSV file into a DataFrame

df = pd.read_csv(input_path+'MatchEvent.csv')
df_riot_item=pd.read_csv(input_path+'riot_item.csv')

df_riot_item=df_riot_item[['itemId','name','explain','tag']]

df['match_id'] = matchId

merged_df = pd.merge(df, df_riot_item, on='itemId', how='left')

merged_df.to_csv(output_path + 'MatchEventItem' + '.csv', index=False)




