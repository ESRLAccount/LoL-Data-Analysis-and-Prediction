import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd


def find_highestdifference(df):
    # Calculate the mean or median values of each column for each group
    mean_group_0 = df[df['win'] == 0].mean()
    mean_group_1 = df[df['win'] == 1].mean()

    # Calculate the absolute differences between the mean values of each column for the two groups
    difference = abs(mean_group_0 - mean_group_1)

    # Sort the differences in descending order to identify columns with the biggest differences
    sorted_difference = difference.sort_values(ascending=False)

    # Select the top five columns with the highest differences
    top_five_columns = sorted_difference.index[:60]

    # Create density plots for the top five columns
    plt.figure(figsize=(12, 8))
    for column in top_five_columns:
        sns.kdeplot(data=df, x=column, hue='win', fill=True, alpha=0.5)

        # Add title and labels
        plt.title('Density Plots for Columns with Highest Differences')
        plt.xlabel(column)
        plt.ylabel('Density')

        # Show legend
        plt.legend(title='Target', labels=['Group 0', 'Group 1'])
        plt.show()

####Read dataset
Inputdata_path = "../DataProcessingProject/data/Output/Masterfile/Matserfile_diffScore.csv"

df = pd.read_csv(Inputdata_path)

find_highestdifference(df)

# Assuming 'column_name' is
# +the column for which you want to create density plots
# and 'target_column' is the name of the binary target variable column (0 or 1)
"""
# Create separate DataFrames for each group (0 and 1)
group_0 = df[df['win'] == 0]
group_1 = df[df['win'] == 1]

cols=['challenges_goldPerMinute','nexusLost','nexusTakedowns']
# Set up the figure and axis
plt.figure(figsize=(10, 6))
for i in range(len(cols)):
    selected_col=cols[i]
    # Plot density plots for each group
    sns.kdeplot(data=group_0[selected_col], label='loss', shade=True)
    sns.kdeplot(data=group_1[selected_col], label='win', shade=True)

    # Add title and labels
    plt.title('Density Plot of Column for Both Groups')
    plt.xlabel(selected_col)
    plt.ylabel('Density')

    # Show legend
    plt.legend()

    # Show plot
    plt.show()
"""