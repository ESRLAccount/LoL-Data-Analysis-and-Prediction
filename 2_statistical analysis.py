import pandas as pd
import numpy as np

from scipy import stats
import seaborn as sns

import matplotlib.pyplot as plt
from sklearn.feature_selection import f_classif, chi2
from sklearn.preprocessing import LabelEncoder, StandardScaler
from scipy.stats import chi2_contingency
from scipy.stats import ttest_ind

def descibe_masterfiledata(filename1,filename2):
    df=pd.read_csv(filename1)
    df_group=df.groupby("tier")['adjustedPoints'].agg(['min','max','size']).reset_index()
    df_group=df_group.sort_values(by='min',ascending=False)
    print(df_group)
    df_group.to_csv(filename2, mode='a', header=True, index=False)
    # no_account=2621
    # df_group['size']=df_group['size']/no_account*100
    # print (df_group)


def SpecificAttributes_TotalAnalysis_VisonPerWin(filename1,filepath,group,Keyword):
    df = pd.read_csv(filename1)


    column_to_drop = 'revisionDate'
    if column_to_drop in df.columns:
        df.drop(columns=[column_to_drop], axis=1, inplace=True)

    plt.figure(figsize=(12, 8))

    # Group by 'Category' and calculate the mean of 'ade' columns
   # grouped_df = df.groupby(group)[pings_columns].mean().reset_index()

    Output_filename = filepath + '_' +  Keyword + 'TotalVision.csv'
    grouped_df = df.groupby(['win'])[vision_columnLists].mean().reset_index()

    grouped_df['Average'] = grouped_df.mean(axis=1)
    grouped_df['sem'] = grouped_df.sem(axis=1)
    grouped_df.to_csv(Output_filename)

    grouped_df = df.groupby(['individualPosition', 'win'])[vision_columnLists].mean().reset_index()
    Output_filename = filepath + '_' + group + '_' + Keyword + 'VisionPerWin.csv'
    grouped_df.to_csv(Output_filename)

    Output_filename = filepath + '_' + group + '_' + Keyword + 'VisionAll.csv'
    grouped_df = df.groupby(['individualPosition'])[vision_columnLists].mean().reset_index()
    grouped_df2 = grouped_df.drop(columns=[group])
    grouped_df['Average'] = grouped_df2.mean(axis=1)
    grouped_df['sem'] = grouped_df2.sem(axis=1)


    grouped_df.to_csv(Output_filename)



    # Melt the DataFrame to long format for easier plotting with seaborn
    # melted_df = grouped_df.melt(id_vars=group, var_name='Metric', value_name='Average')
    # sns.barplot(x=group, y='Average', hue='Metric', data=melted_df)
    if group == 'tier':
        ax = sns.barplot(x=group, y='Average', data=grouped_df, order=ranks_order)
    else:
        ax = sns.barplot(x=group, y='Average', data=grouped_df, color='y', width=0.4)
    # Create a bar plot

    # Adjust the positions of the bars to decrease space between them

    # Add labels and title
    plt.xlabel(group, fontsize=15)
    plt.ylabel(f'Average of {Keyword} related attributes across different {group}', fontsize=14)
    plt.title(f'Average of Columns Containing "{Keyword}" by {group}', fontsize=16, fontweight='bold')

    # Customize x-axis tick labels
    plt.xticks(fontsize=12, fontweight='bold', rotation=45)

    # Show plot
    plt.tight_layout()
    plt.show()

def SpecificAttributes_TotalAnalysis(filename1,filepath,group,Keyword):
    df = pd.read_csv(filename1)

    Output_filename=filepath+'_'+group+'_'+Keyword+'.csv'
    column_to_drop='revisionDate'
    if column_to_drop in df.columns :
        df.drop(columns=[column_to_drop],axis=1,inplace=True)

    plt.figure(figsize=(12, 8))
    # Filter columns with 'ksd' in their name
    pings_columns = [col for col in df.columns if Keyword in col.lower()]
    if Keyword=='vision':
        pings_columns=vision_columnLists

    # Group by 'Category' and calculate the mean of 'ade' columns
    grouped_df = df.groupby(group)[pings_columns].mean().reset_index()


    grouped_df2=grouped_df.drop(columns=[group])
    grouped_df['Average'] = grouped_df2.mean(axis=1)
    grouped_df['sem'] = grouped_df2.sem(axis=1)

    grouped_df.to_csv(Output_filename)
    # Melt the DataFrame to long format for easier plotting with seaborn
    #melted_df = grouped_df.melt(id_vars=group, var_name='Metric', value_name='Average')
    #sns.barplot(x=group, y='Average', hue='Metric', data=melted_df)
    if group == 'tier':
        ax=sns.barplot(x=group, y='Average', data=grouped_df,order=ranks_order)
    else:
        ax=sns.barplot(x=group, y='Average', data=grouped_df,color='y',width=0.4)
    # Create a bar plot

    # Adjust the positions of the bars to decrease space between them


    # Add labels and title
    plt.xlabel(group, fontsize=15)
    plt.ylabel(f'Average of {Keyword} related attributes across different {group}', fontsize=14)
    plt.title(f'Average of Columns Containing "{Keyword}" by {group}', fontsize=16, fontweight='bold')

    # Customize x-axis tick labels
    plt.xticks(fontsize=12, fontweight='bold', rotation=45)

    # Show plot
    plt.tight_layout()
    plt.show()

def Get_RoleLane(filename1,filepath):
    df = pd.read_csv(filename1)
    grouped_df = df.groupby(['individualPosition', 'lane']).size().reset_index(name='counts')
    Output_filename = filepath + '_RoleLane.csv'
    grouped_df.to_csv(Output_filename)
    plt.bar(
        x=[f"{row['individualPosition']}-{row['lane']}" for _, row in grouped_df.iterrows()],
        height=grouped_df['counts']
    )
    # Add title and labels
    plt.title('Counts of Rows Grouped by column1 and column2')
    plt.xlabel('Groups (column1-column2)')
    plt.ylabel('Counts')

    # Rotate x-axis labels for better readability
    plt.xticks(rotation=45)

    # Display the plot
    plt.show()

def SpecificAttributes_DetailAnalysis(filename1,filepath,group,Keyword):

    df = pd.read_csv(filename1)
    column_to_drop='revisionDate'
    if column_to_drop in df.columns :
        df.drop(columns=[column_to_drop],axis=1,inplace=True)
    # Filter columns with 'ksd' in their name

    pings_columns = [col for col in df.columns if Keyword in col.lower()]
    if Keyword=='vision':
        pings_columns.append('wardsPlaced')
    for col in pings_columns:
        df_grouped=df.groupby([group])[col].mean()
        output_filename=filepath + '_'+group+'_'+col+'.csv'
        df_grouped.to_csv(output_filename)


    # Plotting using seaborn
    # Create a figure with subplots
    fig, axes = plt.subplots(3, 2, figsize=(20, 20))  # 4 rows, 3 columns for 12 plots
    # Flatten the axes array for easy iteration
    axes = axes.flatten()
    colors = sns.color_palette('husl', n_colors=len(pings_columns))
    for i, (col, color) in enumerate(zip(pings_columns, colors)):
        if group == 'tier':
            sns.barplot(x=group, y=col, data=df, errorbar=None, palette=[color], label=col, order=ranks_order,ax=axes[i])
        else:
            sns.barplot(x=group, y=col, data=df, errorbar=None, palette=[color], label=col, ax=axes[i])

        axes[i].set_title(col, fontsize=20,fontweight='bold')
        axes[i].set_xlabel(group,fontsize=8,fontweight='bold')
        axes[i].set_ylabel('Frequency',fontweight='bold')
        axes[i].tick_params(axis='x', labelsize=10)  # Adjust font size if needed
        axes[i].set_xticklabels(axes[i].get_xticklabels(), fontweight='bold',fontsize=15,rotation=30)
    plt.tight_layout()
    plt.show()
    # Filter columns with 'ksd' in their name and calculate mean, min, max using lambda function


def AllAttributes_DetailAnalysis(filename1,filename2,group,collist):
    def is_numeric(column):
        return pd.to_numeric(column, errors='coerce').notnull().all()

    df = pd.read_csv(filename1)
    nrows=int(len(collist)/4)
    # Create a figure with subplots
    fig, axes = plt.subplots(nrows, 4, figsize=(nrows*10, 100))  # 4 rows, 3 columns for 12 plots
    # Flatten the axes array for easy iteration
    axes = axes.flatten()
    colors = sns.color_palette('husl', n_colors=len(collist))
    for i, (col, color) in enumerate(zip(collist, colors)):
        #if is_numeric(df[col]):
            if group == 'tier':
                sns.barplot(x=group, y=col, data=df, errorbar=None, palette=[color], label=col, order=ranks_order,ax=axes[i])
            else:
                sns.barplot(x=group, y=col, data=df, errorbar=None, palette=[color], label=col, ax=axes[i])

            axes[i].set_title(col, fontsize=20,fontweight='bold')
            axes[i].set_xlabel(group,fontsize=8,fontweight='bold')
            axes[i].set_ylabel('Frequency',fontweight='bold')
            axes[i].tick_params(axis='x', labelsize=10)  # Adjust font size if needed
            axes[i].set_xticklabels(axes[i].get_xticklabels(), fontweight='bold',fontsize=15,rotation=30)
    plt.tight_layout()
    plt.show()
    # Filter columns with 'ksd' in their name and calculate mean, min, max using lambda function



def outlierremoval(file):
    df = pd.read_csv(file)
    # Specify the column and threshold for filtering
    column_to_filter = 'gameDuration'
    threshold = 900

    # Filter rows based on the condition
    filtered_df = df[df[column_to_filter] >= threshold]

    return filtered_df

def corelation_analysis(file):
    df = pd.read_csv(file)
    # Assuming df is your DataFrame with 'rank' and 'win/loss' columns
    # Calculate Point-Biserial Correlation between 'rank' and 'win/loss'
    point_biserial_corr, _ = stats.pointbiserialr(df['Rank'], df['win'])

    # Add the Point-Biserial Correlation coefficient to your DataFrame
    df['point_biserial_corr'] = point_biserial_corr

def create_densityplot(file):
    df = pd.read_csv(file)

    sns.set(style="whitegrid")
    palette = sns.color_palette("husl", n_colors=len(df['Rank'].unique()))

    # Create the density plot
    plt.figure(figsize=(12, 8))

    ax = sns.barplot(
        x="Rank",
        y="percent",
        data=df,

        ci='sd',
        capsize=.07)
    # Set title
    plt.title("Barplot example with tips dataset")

    #sns.kdeplot(data=df, x='Rank', y='percent', fill=True, palette=palette, common_norm=False)
    #sns.scatterplot(x="Rank", y="percent", data=df)
    #sns.kdeplot(data=df, x='Rank', fill=True,palette=palette)
    plt.title('Density Plot of Percentage Distribution within Ranks')
    plt.xlabel('Percentage')
    plt.ylabel('Density')
    plt.legend(title='Rank')
    plt.show()

def Chi_SquareTest_Win(filename):

    df = pd.read_csv(filename)

   # df['tier'] = np.random.choice([f'Category_{i}' for i in range(10)], 100)

    # Convert boolean target to integer for analysis
    df['target_encoded'] = df['win'].astype(int)

    y = df['target_encoded']
    df = df.select_dtypes(include='number')
    # Separate features and target

    X=df

    # Handle missing values
    X_filled = X.copy()
    for col in X.columns:
        if X[col].dtype in [np.float64, np.int64]:  # Continuous feature
            X_filled[col].fillna(X[col].mean(), inplace=True)
        else:  # Categorical feature
            X_filled[col].fillna(X[col].mode()[0], inplace=True)

    # Standardize features for ANOVA
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_filled)

    # Perform ANOVA for continuous features
    f_values, p_values = f_classif(X_scaled, y)

    # Perform Chi-Square test for categorical features (if any)
    chi2_p_values = []
    for col in X.columns:
        if df[col].dtype == 'object' or df[col].nunique() < 10:  # Assuming categorical if unique values < 10
            contingency_table = pd.crosstab(df[col], y)
            chi2_stat, p, dof, ex = chi2_contingency(contingency_table)
            chi2_p_values.append((col, p))

    # Combine ANOVA and Chi-Square results
    anova_results = pd.DataFrame({
        'Feature': X.columns,
        'p-value': p_values
    })
    chi2_results = pd.DataFrame(chi2_p_values, columns=['Feature', 'p-value'])

    # Concatenate results
    combined_results = pd.concat([anova_results, chi2_results]).sort_values(by='p-value',ascending=False)

    # Set a p-value threshold
    p_value_threshold = 0.05

    # Filter out features based on the p-value threshold
    filtered_results = combined_results[combined_results['p-value'] < p_value_threshold]

    top_20_results = filtered_results.head(20)
    # Plot the p-values
    plt.figure(figsize=(14, 10))
    plt.barh(top_20_results['Feature'], top_20_results['p-value'], color='skyblue')
    plt.xlabel('p-value')
    plt.ylabel('Feature')
    plt.title('P-values of Features from ANOVA and Chi-Square Tests')
    plt.gca().invert_yaxis()
    plt.show()

    # Save results to CSV
    combined_results.to_csv('feature_dependence_results_win.csv', index=False)

def Chi_SquareTest_Rank(filename):

        df = pd.read_csv(filename)

        # df['tier'] = np.random.choice([f'Category_{i}' for i in range(10)], 100)

        # Encode target variable
        label_encoder = LabelEncoder()
        df['target_encoded'] = label_encoder.fit_transform(df['tier'])
        df = df.select_dtypes(include='number')
        # Separate features and target
        X = df.drop(columns=['target_encoded'])
        y = df['target_encoded']

        # Handle missing values
        X_filled = X.copy()
        for col in X.columns:
            if X[col].dtype in [np.float64, np.int64]:  # Continuous feature
                X_filled[col].fillna(X[col].mean(), inplace=True)
            else:  # Categorical feature
                X_filled[col].fillna(X[col].mode()[0], inplace=True)

        # Standardize features for ANOVA
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X_filled)

        # Perform ANOVA for continuous features
        f_values, p_values = f_classif(X_scaled, y)

        # Perform Chi-Square test for categorical features (if any)
        chi2_p_values = []
        for col in X.columns:
            if df[col].dtype == 'object' or df[col].nunique() < 10:  # Assuming categorical if unique values < 10
                contingency_table = pd.crosstab(df[col], df['target_encoded'])
                chi2_stat, p, dof, ex = chi2_contingency(contingency_table)
                chi2_p_values.append((col, p))

        # Combine ANOVA and Chi-Square results
        anova_results = pd.DataFrame({
            'Feature': X.columns,
            'p-value': p_values
        })
        chi2_results = pd.DataFrame(chi2_p_values, columns=['Feature', 'p-value'])

        # Concatenate results
        combined_results = pd.concat([anova_results, chi2_results]).sort_values(by='p-value', ascending=False)
        # Set a p-value threshold
        p_value_threshold = 0.05

        # Filter out features based on the p-value threshold
        filtered_results = combined_results[combined_results['p-value'] < p_value_threshold]

        top_20_results = filtered_results.head(20)
        # Plot the p-values
        plt.figure(figsize=(14, 10))
        plt.barh(top_20_results['Feature'], top_20_results['p-value'], color='skyblue')
        plt.xlabel('p-value')
        plt.ylabel('Feature')
        plt.title('P-values of Features from ANOVA and Chi-Square Tests')
        plt.gca().invert_yaxis()
        plt.show()

        # Save results to CSV
        combined_results.to_csv('feature_dependence_results.csv', index=False)


inputdata_path1 = "../data/OutputRank/MatchResume/FinalRankedMatchResume_Masterfile.csv"

inputdata_pathTeam = "../data/OutputRank/MatchResume/FinalTeamMatchResume_Masterfile.csv"

inputdata_pathR = "../data/OutputRank/MatchResume/FinalMatchResume_Masterfile.csv"

inputdata_path2 = "../../RiotProject/models/SummonerIdFromNames.csv"

inputdata_path3 = "../data/OutputRank/MatchResume/MatchResume_MasterfilewithRank2.csv"

outputdata_path1 = "../../../RiotProject/models/SummonerNames_witnoutRank2.csv"

outputdata_path2 = "../data/OutputRank/MatchResume/StatisticalAnalysis.csv"
outputdata_path3= "../data/OutputRank/ReadyToPlot/StatisticalAnalysis_"

ranks_order = ["IRON", "BRONZE", "SILVER", "GOLD", "PLATINUM", "EMERALD", "DIAMOND", "MASTER", "GRANDMASTER",
               "CHALLENGER"]

vision_columnLists=['visionScore','enemyVisionPings','wardsPlaced']

inputdata_pathforSelectedCols = "../data/Input/MatchResume_FinalColsRanked.csv"
df_cols = pd.read_csv(inputdata_pathforSelectedCols)

column_list = df_cols['feature_name'].tolist()

inputdata_pathforSelectedCols = "../data/Input/MatchResume_FinalCols.csv"
df_colsrole = pd.read_csv(inputdata_pathforSelectedCols)

column_listrole = df_colsrole['feature_name'].tolist()

#Get_RoleLane(inputdata_path1,outputdata_path3)
#describe_results=descibe_masterfiledata(inputdata_path1,outputdata_path2)

SpecificAttributes_TotalAnalysis(inputdata_pathTeam,outputdata_path3,'win','vision')

#SpecificAttributes_TotalAnalysis(inputdata_path1,outputdata_path3,'tier','pings')
#SpecificAttributes_TotalAnalysis(inputdata_path1,outputdata_path3,'tier','vision')
SpecificAttributes_TotalAnalysis_VisonPerWin(inputdata_path1,outputdata_path3,'individualPosition','vision')
#SpecificAttributes_TotalAnalysis(inputdata_path1,outputdata_path3,'individualPosition','pings')

#SpecificAttributes_DetailAnalysis(inputdata_path1,outputdata_path3,'tier','pings')
#SpecificAttributes_DetailAnalysis(inputdata_path1,outputdata_path3,'tier','vision')
#SpecificAttributes_DetailAnalysis(inputdata_pathR,outputdata_path3,'individualPosition','vision')

#AllAttributes_DetailAnalysis(inputdata_path1,outputdata_path3,'tier',column_list)
#AllAttributes_DetailAnalysis(inputdata_path1,outputdata_path3,'individualPosition',column_list)
#Chi_SquareTest_Rank(inputdata_path1)
#Chi_SquareTest_Win(inputdata_path1)
"""
df = pd.read_csv(inputdata_path3,usecols=['Rank','summonerName','riotIdTagline'])
#df=df[df['Rank'].isnull()]
df['riotIdTagline']=df['riotIdTagline'].fillna('')
df['tagedsummonerName'] = df['summonerName'] + "#" + df['riotIdTagline']
unique_combinations = df.drop_duplicates(subset=['tagedsummonerName'])
unique_combinations.to_csv(outputdata_path1)
"""
""" 

df = pd.read_csv(inputdata_path1)
df_group=df.groupby("Rank").agg({'LeaguePoints':['min','max','size']}).reset_index()


df = pd.read_csv(inputdata_path2)
df_group=df.groupby("Rank").agg({'LeaguePoints':['min','max','size']}).reset_index()

df_group = df_group.rename(columns={'size': 'valuesize'}).reset_index()

#f_group['perecnt']=df_group['valuesize']/2621
#df_group.to_csv(outputdata_path2,index=False)
print (df_group)




"""
##finding minimum game length


#create_densityplot(outputdata_path2)
# Display the DataFrame with the added Point-Biserial Correlation

#corelation_analysis(inputdata_path1)

##outlier removal:
##df=outlierremoval(inputdata_path1)
##df = df.rename(columns={'tier': 'Rank','rank':'T_rank'})
##print(len(df))
##df.to_csv(outputdata_path3,index=False)
