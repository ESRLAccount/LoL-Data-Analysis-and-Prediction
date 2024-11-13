
### creating a masterfile including all matchchallenge datarows

import pandas as pd

import shutil
import os
import glob
from datetime import datetime
from difflib import SequenceMatcher
import time

#inputMatchResume_path = "../../LOLDataset/MatchResume_EUW/*.csv"

inputMatchResume_path = "../../RiotProject/data/OutputMatchResume1"#/*.csv"
outputMatchResume_path = "../data/OutputRank/MatchResume/MatchResume_Masterfile1.csv"


inputdata_pathforSelectedCols = "../data/Input/MatchResume_SelectedCols.csv"
df_cols = pd.read_csv(inputdata_pathforSelectedCols)

# Convert the DataFrame column to a Python array
selected_array = df_cols.iloc[:, 0].tolist()

csv_files=[]
## Assuming all your CSV files are in the same directory
#csv_files = glob.glob(inputMatchResume_path)

for root, dirs, files in os.walk(inputMatchResume_path):
    for file in files:
        if file.endswith('.csv'):
            csv_files.append(os.path.join(root, file))

#files_limit =6
files_processed = 0

dfs=[]
combined_df = pd.DataFrame()
# Iterate through each CSV file and append its data to the combined DataFrame
for csv_file in csv_files:
    # Read the CSV file into a DataFrame
    #print (csv_file)
    df = pd.read_csv(csv_file)#,usecols=selected_array)
    if len(df)>0:
      if 'gameDuration' in df.columns:
        if df['gameDuration'].iloc[0]>=900 :
        # Select columns without an underscore

            if  set(selected_array).issubset(df.columns):

            #if len(Rootcolumns)==len(selected_array):
                #######
                subset_df = df[selected_array]

                timestamp_ms = df.iloc[0]['gameStartTimestamp']

                timestamp_seconds = timestamp_ms / 1000  # Convert milliseconds to seconds
                formatted_time = datetime.utcfromtimestamp(timestamp_seconds).strftime('%Y_%m_%d-%H_%M_%S')
                subset_df.insert(0, 'game_date', formatted_time)
                ##########################

                dfs.append(subset_df)

                files_processed += 1
                ##if files_processed >= files_limit:
                ##    break
                print (files_processed)
            else :
                print(csv_file)
            #   missing_columns = set(selected_array) - set(df.columns)
        # Increment the counter


    # Break the loop when the desired number of files is reached


# Concatenate all DataFrames in the list into a single DataFrame
combined_df = pd.concat(dfs, ignore_index=True)
combined_df.to_csv(outputMatchResume_path, index=False)

######################################################
