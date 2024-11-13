import pandas as pd
import matplotlib.pyplot as plt

# Load game data from CSV into a DataFrame
df = pd.read_csv('data\\OutputRank\\MatchTimeline\\MatchTimeline_masterfile_withRole.csv')  # Update the path to the correct CSV file if necessary


if 'role' not in df.columns or 'championName' not in df.columns:
    raise ValueError("The DataFrame must contain 'role' and 'championName' columns.")


roles_of_interest = ['CARRY', 'DUO', 'NONE', 'SOLO']


most_popular_champions = {}


for role in roles_of_interest:
    # Filter the DataFrame for the current role
    role_df = df[df['role'] == role]

    # Group by 'championName' and count occurrences
    champion_counts = role_df['championName'].value_counts().reset_index(name='count')
    champion_counts.columns = ['championName', 'count']

    # Store the result in the dictionary
    most_popular_champions[role] = champion_counts

# Print the most popular champions for each role
for role, champions in most_popular_champions.items():
    print(f"Most Popular Champions in {role}:")
    print(champions.head())  # Display the top champions for each role
    print()

# Plotting the results for each role
for role, champions in most_popular_champions.items():
    plt.figure(figsize=(12, 6))
    plt.bar(champions['championName'], champions['count'], color='skyblue')
    plt.xlabel('Champion Name')
    plt.ylabel('Count')
    plt.title(f'Most Popular Champions in {role}')
    plt.xticks(rotation=90)  # Rotate champion names for better readability
    plt.tight_layout()
    plt.show()
