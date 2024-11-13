###create rowscore & diffscore for matchResume and preparing the masterfile for deeplearning
import pandas as pd
from sklearn.feature_selection import VarianceThreshold
import matplotlib.pyplot as plt
import seaborn as sns

def get_lowVarienceCols(df):
    # Specify the variance threshold
    variance_threshold = 0.01  # You can adjust this threshold based on your preference

    # Separate features and target variable
    X = df.drop('win', axis=1)

    variances = X.var()
    low_variance_columns = variances[variances < variance_threshold].index


    df[low_variance_columns].to_csv("data/Output/MatchChallenges/featurenotvarienceEU.csv")
    # Display selected features

    df=df.drop(columns=low_variance_columns,axis=1)
    return (df)

def outlierremoval(df):
    # Specify the column and threshold for filtering
    column_to_filter = 'gameDuration'
    threshold = 900

    # Filter rows based on the condition
    filtered_df = df[df[column_to_filter] >= threshold]

    return filtered_df

def Normalize_perGameduration (df):
    # Define the column names to exclude from normalization
    exclude_cols = ['win']  # Add any column names you want to exclude

    df = df.apply(pd.to_numeric, errors='coerce')

    #df['gameDuration'] = pd.to_numeric(df['gameDuration'], errors='coerce')

    # Get the list of columns to normalize (all columns except exclude_cols)
    normalize_cols = df.columns.difference(exclude_cols)
    normalized_df = df.copy()

    # Normalize all columns based on the game duration
    normalized_df[normalize_cols] = normalized_df[normalize_cols].div(normalized_df['gameDuration'], axis=0)

    return(normalized_df)

def get_diffrencescore(df):
    diff_df = pd.DataFrame()
    suffix = 'diff'
    # Iterate over rows in the original DataFrame (excluding the last row)
    #for i in range(10):
    for i in range(len(df) - 1):
        differences=[]
        print (i)
        # Check if the row index is odd or even
        if i % 2 == 0:  # Even index
            differences = df.iloc[i] - df.iloc[i+1]

        else:  # Odd index
            differences = df.iloc[i] - df.iloc[i-1]


        # Rename the differences columns
        differences = differences.add_suffix(f'_{suffix}')

        # Append the original row and the differences as new columns
        diff_df = pd.concat([diff_df, pd.concat([df.iloc[i], differences])], axis=1)

    # Transpose the resulting DataFrame to have rows and columns in the desired order
    diff_df = diff_df.transpose()

    # Reset the index of the new DataFrame
    diff_df.reset_index(drop=True, inplace=True)
    diff_df.drop(['win_diff'], axis=1, inplace=True)

    return diff_df


def get_masterfilestats(filename):
    df = pd.read_csv(filename)
    print('number of matches', len(df) / 2)

    print('number of features', len(df.columns))

    print(df['gameDuration'].describe())

    # Create the box plot
    plt.figure(figsize=(10, 6))
    sns.boxplot(data=df, x='gameDuration')
    plt.title('Box Plot of Attribute')
    plt.xlabel('Attribute')
    plt.show()



Inputdata_path =  "data/OutputRank/MatchResume/MatchResume_Masterfile.csv"
inputpath2 = "data/OutputRank/MatchTimeline/MatchTimeline_PerMatchPhase.csv"

df=pd.read_csv(inputpath2,nrows=1000)


outputdata_pathRowScore = "data/OutputRank/MatchResume/Masterfile_RowScoreAfterGamelenNormalization.csv"
outputdata_pathdiffScore = "data/OutputRank/MatchResume/Masterfile_diffScoreAfterGamelenNormalization.csv"



#get_masterfilestats(Inputdata_path)

df=pd.read_csv(Inputdata_path)

#filtered_df=outlierremoval(df)

#filtered_df=Normalize_perGameduration(df)


#filtered_df.drop(['gameStartTimestamp','gameDuration'],axis=1,inplace=True)

#reduced_df=get_lowVarienceCols(filtered_df)

#filtered_df.to_csv(outputdata_pathRowScore, index=False)
# Initialize the MinMaxScaler

filtered_df = filtered_df.select_dtypes(include='number')



diff_df=get_diffrencescore(df)

diff_df.to_csv(outputdata_pathdiffScore, index=False)

#print(len(filtered_df.columns),len(reduced_df.columns))

#print (reduced_df['challenges_gameLength'].max(),reduced_df['challenges_gameLength'].min(),reduced_df['challenges_gameLength'].mean(),reduced_df['challenges_gameLength'].std())

