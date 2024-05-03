#!/usr/bin/env python
# coding: utf-8

# In[325]:


import os
import pandas as pd


# In[326]:


df = pd.read_csv(r"/Users/danielbrown/Desktop/Golfer_Dataframe_total.csv")
df_1 = pd.read_csv(r"/Users/danielbrown/Desktop/combined_df_clusters_.csv")

# Assuming df is your DataFrame
df_1 = df_1.dropna()

df_1['Hole'] = df_1['Hole'].astype(int)
df_1['Hole_Par'] = df_1['Hole_Par'].astype(int)
#df_1


# In[327]:


# Assuming your DataFrame is named df
df.rename(columns={'Event Title': 'Event'}, inplace=True)

df


# In[328]:


# Convert 'Event Date' column to datetime format
df['Event Date'] = pd.to_datetime(df['Event Date'])

# Extract year from 'Event Date' column
df['Year'] = df['Event Date'].dt.year
#df


# In[329]:


players = [
    "Justin Hastings (a)", "Jackson Van Paris (a)", "Nick Dunlap",
    "Daniel Berger", "Evan Harmeling", "Wilson Furr", "Alejandro Tosti"
]

# Filter the DataFrame
players_df = df[df["Player Name"].isin(players)]

# Save the filtered DataFrame to a CSV file
players_df.to_csv("/Users/danielbrown/Desktop/a_few_golfers.csv", index=False)


# In[ ]:





# In[330]:


#We need to temporarily remove "The American Express Tournament", "PGA Tour Q School Korn Ferry", "The RSM Classic"
#The reason is because these events play on multiple courses and that messes with my hole type probabilities
#In the future I will correct this, but for now lets just remove these tournaments

events = [
    "The American Express", "The RSM Classic", "Farmers Insurance Open"
]

# Filter the DataFrame
events_df = df[df["Event"].isin(events)]

# Save the filtered DataFrame to a CSV file
#players_df.to_csv("/Users/danielbrown/Desktop/a_few_golfers.csv", index=False)
#events_df

# Assuming df is your main DataFrame and filtered_df is the subset you've created

# Identify the indices of rows in filtered_df
indices_to_remove = events_df.index

# Drop these indices from df to remove the rows
df = df.drop(indices_to_remove)

# Now df contains only the entries that were not in filtered_df


# In[331]:


df


# In[332]:


# Group by Event and Year and get the original event date for each group
original_dates = df[df['Round'] == 1].groupby(['Event', 'Year'])['Event Date'].first().reset_index()

# Merge original dates back to the DataFrame
df = pd.merge(df, original_dates, on=['Event', 'Year'], suffixes=('', '_original'))

# Calculate modified dates
df['Modified Date'] = df['Event Date_original'] + pd.to_timedelta(df['Round'] - 1, unit='D')

# Display the DataFrame with the new 'Modified Date' column
#print(df[['Event', 'Year', 'Round', 'Modified Date']])
#df.head(50)


# In[333]:


df


# In[334]:


df['FPTS_EBPB'] = 13*df['Double_Eagle_or_Better'] + 8*df['Eagle'] + 3*df["Birdie"] + 0.5*df['Par'] + (-0.5)*df['Bogey'] + (-1)*df['Double_Bogey'] + (-1)*df['Worse_than_Double_Bogey'] + 5*df['Hole_in_1'] #+ 3*df['Streak_of_3_Birdies_or_Better'] + 3*df['Bogey_Free_Round'] + 5*df['Under_70_All_Rounds']
df.to_csv(r"/Users/danielbrown/Desktop/Golfer_Every_hole.csv")


# In[335]:


# Display basic info about the DataFrame
#print("DataFrame Info:")
#print(df.info())

# Count NaN values for each column
#print("\nCount of NaN Values:")
#print(df.isna().sum())

# Convert non-numeric values in 'Score' and 'Scores' columns to NaN
df['Score'] = pd.to_numeric(df['Score'], errors='coerce')
df['Scores'] = pd.to_numeric(df['Scores'], errors='coerce')

# Calculate summary statistics for numeric columns
#print("\nSummary Statistics:")
#print(df.describe())

#Filter by event date. Filter for the last 8 months
import pandas as pd

def filter_last_x_months(df, x):
    """
    Filter the DataFrame for the last X months of data based on the 'Event Date' column.

    Parameters:
    df (DataFrame): The DataFrame to filter.
    x (int): The number of months to filter the data for.

    Returns:
    DataFrame: The filtered DataFrame.
    """
    # Ensure the 'Event Date' column is in datetime format
    df['Event Date'] = pd.to_datetime(df['Event Date'])

    # Get the current date
    current_date = pd.to_datetime('today')

    # Calculate the date x months ago from the current date
    x_months_ago = current_date - pd.DateOffset(months=x)

    # Filter the DataFrame for rows where the 'Event Date' is within the last x months
    filtered_df = df[df['Event Date'] >= x_months_ago]

    return filtered_df

filtered_df = filter_last_x_months(df, 6)
df = filtered_df


# In[336]:


df


# In[337]:


# Assuming df and df_1 are your dataframes
# Perform the left join on Event and Hole columns
merged_df = pd.merge(df_1, df, on=['Event', 'Hole'], how='left')

# Assuming df_2 is your third dataframe to be joined
# Perform the left join on the merged dataframe and df_2
#final_merged_df = pd.merge(merged_df, df_2, on='common_column', how='left')
merged_df = merged_df[['Player Name','Event Date','FPTS_EBPB','Cluster']]

merged_df


# In[338]:


# Keep the last occurrence of each duplicated row
#merged_df = merged_df.drop_duplicates(keep='last')
#merged_df


# In[339]:


merged_df.to_csv(r"/Users/danielbrown/Desktop/Golfer_Dataframe_lets_take_aloook.csv")


# In[ ]:





# In[340]:


# Function to calculate percentage of each fantasy point value for a player in a cluster
def calculate_percentage(df):
    total_count = df.shape[0]
    points_count = df['FPTS_EBPB'].value_counts(normalize=True) * 100
    points_count['Data Points'] = total_count  # Add 'Data Points' entry
    return points_count

# Group by Player Name and Cluster, then apply the percentage calculation function
player_cluster_percentage = merged_df.groupby(['Player Name', 'Cluster']).apply(calculate_percentage).unstack(fill_value=0)

# Reset index to flatten the DataFrame
player_cluster_percentage.reset_index(inplace=True)

# Rename columns for clarity
player_cluster_percentage.columns.name = None

# Round the percentages to 3 decimal points and convert them to decimals
player_cluster_percentage.iloc[:, 2:-1] = player_cluster_percentage.iloc[:, 2:-1].apply(lambda x: round(x / 100, 3))

# Print the modified DataFrame
#print(player_cluster_percentage)

print(player_cluster_percentage)


# In[341]:


player_cluster_percentage.to_csv(r"/Users/danielbrown/Desktop/player_distributions.csv")


# In[342]:


# Display basic info about the DataFrame
print("DataFrame Info:")
print(player_cluster_percentage.info())

# Count NaN values for each column
#print("\nCount of NaN Values:")
#print(df.isna().sum())

# Calculate summary statistics for numeric columns
#print("\nSummary Statistics:")
#print(player_cluster_percentage.describe())


# In[343]:


# Create an empty DataFrame to store the top 10 players for each cluster
top_10_players_df = pd.DataFrame(columns=player_cluster_percentage.columns)

# Iterate over each cluster
for cluster in player_cluster_percentage['Cluster'].unique():
    # Filter clusters where Data Points > 50
    filtered_cluster_df = player_cluster_percentage[(player_cluster_percentage['Cluster'] == cluster) & (player_cluster_percentage['Data Points'] > 50)]
    
    if not filtered_cluster_df.empty:
        # Sort the filtered DataFrame by '3.0' column in descending order within each cluster
        sorted_cluster_df = filtered_cluster_df.sort_values(by=[3.0], ascending=False)
        
        # Select top 10 players based on '3.0' column
        top_10_players_cluster = sorted_cluster_df.head(10)
        
        # Append the top 10 players of the cluster to the top_10_players_df DataFrame
        top_10_players_df = pd.concat([top_10_players_df, top_10_players_cluster])

# Reset index for better display
top_10_players_df.reset_index(drop=True, inplace=True)

# Print the top 10 players for each cluster with the highest '3.0' number and Data Points > 50
print(top_10_players_df)


# In[344]:


top_10_players_df.to_csv(r"/Users/danielbrown/Desktop/top_10_players_df.csv")


# In[345]:


print(player_cluster_percentage.columns)


# In[346]:


# Group the DataFrame by 'Cluster' and count the number of unique players with over 100 data points in each group
players_over_100_datapoints = player_cluster_percentage[player_cluster_percentage['Data Points'] > 50].groupby('Cluster')['Player Name'].nunique()

# Print the result
print(players_over_100_datapoints)


# In[347]:


# Get a list of players who have over 100 data points for all clusters
players_over_100_all_clusters = player_cluster_percentage.groupby('Player Name').filter(lambda x: (x['Data Points'] > 50).all())['Player Name'].unique()

# Count the number of players
num_players_over_100_all_clusters = len(players_over_100_all_clusters)

# Print the result
print("Number of players with over 100 data points for all clusters:", num_players_over_100_all_clusters)


# In[348]:


# Filter the DataFrame based on players with over 100 data points for all clusters
filtered_players_df = player_cluster_percentage[player_cluster_percentage['Player Name'].isin(players_over_100_all_clusters)]

# Reset the index
filtered_players_df.reset_index(drop=True, inplace=True)

# Print the filtered DataFrame
print(filtered_players_df)
filtered_players_df.to_csv(r"/Users/danielbrown/Desktop/over_100_all_clusters.csv")


# In[349]:


df = filtered_players_df
import numpy as np


# In[350]:


# Function to simulate a round of golf
def simulate_round(probabilities, points):
    return np.random.choice(points, p=probabilities)

# Function to get user input for number of holes for each cluster
def get_holes_input():
    holes = {}
    clusters = ['3-0', '3-1', '4-0', '4-1', '4-2', '5-0', '5-1']
    for cluster in clusters:
        count = int(input(f"How many holes of cluster {cluster} are there? "))
        holes[cluster] = count
    return holes

# Function to get the order of holes from the user
def get_holes_order(holes):
    while True:
        order_str = input("Enter the order of the clusters separated by commas (e.g., 3-0,4-1,3-0): ")
        order = order_str.split(',')
        # Validate the order
        valid = True
        temp_holes = holes.copy()
        for cluster in order:
            if cluster not in temp_holes or temp_holes[cluster] == 0:
                valid = False
                break
            temp_holes[cluster] -= 1
        if valid and all(count == 0 for count in temp_holes.values()):
            return order
        print("Invalid order. Please ensure the order matches the number of holes for each cluster.")

# Get user input for number of holes for each cluster
holes = get_holes_input()

# Get the order of holes from the user
order = get_holes_order(holes)

print("Ordered holes:", order)



# Points earned for each type of hole
points = [-1.0, -0.5, 0.5, 3.0, 8.0, 13.0]

# Simulate a round of golf for each player
for player_name, player_data in df.groupby('Player Name'):
    total_points = 0
    for _, row in player_data.iterrows():
        cluster = row['Cluster']
        count = holes.get(cluster, 0)
        probabilities = row[points].values.tolist()  # Select only the probabilities for the points
        probabilities /= np.sum(probabilities)  # Normalize probabilities to sum up to 1
        for _ in range(count):
            total_points += simulate_round(probabilities, points)
    print(f"{player_name} scored {total_points} points.")


# In[351]:


#4-2,5-0,4-2,3-1,4-1,3-1,4-1,5-0,4-1,4-1,4-2,3-0,5-0,4-2,5-0,3-0,4-1,4-1


# In[352]:


# Function to convert points to strokes and accumulate them
def convert_points_to_strokes_and_accumulate(round_data):
    point_to_strokes = {-1: 2, -0.5: 1, 0.5: 0, 3: -1, 8: -2, 13: -3}
    strokes_round_data = round_data.copy()
    strokes_round_data['Cumulative Strokes'] = 0

    for index, row in strokes_round_data.iterrows():
        cumulative_strokes = 0
        for col in row.index:
            if col not in ['Player Name', 'Total Points', 'Cumulative Strokes']:
                strokes = point_to_strokes.get(row[col], 0)  # Default to 0 if not found
                cumulative_strokes += strokes
                strokes_round_data.at[index, col] = cumulative_strokes
        strokes_round_data.at[index, 'Cumulative Strokes'] = cumulative_strokes

    return strokes_round_data


def convert_points_to_strokes_and_accumulate(round_data):
    point_to_strokes = {-1: 2, -0.5: 1, 0.5: 0, 3: -1, 8: -2, 13: -3}
    strokes_round_data = round_data.copy()
    strokes_round_data['Cumulative Strokes'] = 0

    for index, row in strokes_round_data.iterrows():
        cumulative_strokes = 0
        for col in row.index:
            if col not in ['Player Name', 'Total Points', 'Cumulative Strokes']:
                strokes = point_to_strokes.get(row[col], 0)  # Default to 0 if not found
                cumulative_strokes += strokes
                strokes_round_data.at[index, col] = cumulative_strokes
        strokes_round_data.at[index, 'Cumulative Strokes'] = cumulative_strokes

    return strokes_round_data


def apply_bogey_free_round_bonus(round_df):
    for index, row in round_df.iterrows():
        if all(score >= 0 for score in row[1:-1]):  # Check if all scores are non-negative
            round_df.at[index, 'Total Points'] += 3  # Add 3 points for a Bogey Free Round
    return round_df

def apply_bonus_points(round_df):
    bonus_points_info = []
    for player_name in round_df['Player Name'].unique():
        player_data = round_df[round_df['Player Name'] == player_name].iloc[0]
        scores = player_data.drop(labels=['Player Name', 'Total Points']).values
        for i in range(len(scores) - 2):
            if scores[i] == 3.0 and scores[i + 1] == 3.0 and scores[i + 2] == 3.0:
                round_df.loc[round_df['Player Name'] == player_name, f'Hole {i + 3}'] += 3  # Bonus for third consecutive hole
                bonus_points_info.append({'player_name': player_name, 'hole': i + 3})
                break  # Only one bonus per round per player
    round_df['Total Points'] = round_df.iloc[:, 1:-1].sum(axis=1)  # Recalculate total points after bonuses
    return round_df, bonus_points_info


# In[353]:


#This cuts according to Fantasy Points,it works. It cuts half the field
def simulate_multiple_4_rounds_with_cut_line(df, holes, points, num_simulations=2):
    cumulative_totals_list = []
    all_rounds_data_list = []

    for simulation_num in range(1, num_simulations + 1):
        rounds_data = []
        players_not_making_cut = set()

        for round_num in range(1, 5):
            round_scores = pd.DataFrame(columns=['Player Name'] + [f'Hole {i+1}' for i in range(sum(holes.values()))])

            for player_name, player_data in df.groupby('Player Name'):
                if round_num > 2 and player_name in players_not_making_cut:
                    continue

                player_points = []
                for _, row in player_data.iterrows():
                    cluster = row['Cluster']
                    count = holes.get(cluster, 0)
                    probabilities = row[points].values.tolist()
                    probabilities /= np.sum(probabilities)
                    for _ in range(count):
                        player_points.append(simulate_round(probabilities, points))

                round_scores.loc[len(round_scores)] = [player_name] + player_points
                
            round_strokes = convert_points_to_strokes_and_accumulate(round_scores)
            round_scores['Total Points'] = round_scores.iloc[:, 1:-1].sum(axis=1)
            round_scores, _ = apply_bonus_points(round_scores)  # Apply consecutive three bonus
            round_scores = apply_bogey_free_round_bonus(round_scores)  # Apply Bogey Free Round bonus
            rounds_data.append(round_scores)

            if round_num == 2:
                cumulative_total = pd.concat(rounds_data).groupby('Player Name')['Total Points'].sum().reset_index()
                cumulative_total_sorted = cumulative_total.sort_values(by='Total Points', ascending=False)
                num_players_to_cut = len(cumulative_total_sorted) // 2
                players_to_cut = set(cumulative_total_sorted['Player Name'].tail(num_players_to_cut).values)
                players_not_making_cut.update(players_to_cut)

        all_rounds_data_list.append(rounds_data)
        cumulative_total = pd.concat(rounds_data).groupby('Player Name')['Total Points'].sum().reset_index()
        cumulative_total_sorted = cumulative_total.sort_values(by='Total Points', ascending=False)
        cumulative_totals_list.append(cumulative_total_sorted)

    return cumulative_totals_list, all_rounds_data_list

# Example usage:
df_sims = simulate_multiple_4_rounds_with_cut_line(df, holes, points, num_simulations=500)
print(df_sims)


# In[354]:


#This cuts according to Fantasy Points,it works. It cuts the amount of players you say to. It is an input

'''

import pandas as pd
import numpy as np

def simulate_multiple_4_rounds_with_cut_line(df, holes, points, num_players_to_cut, num_simulations=2):
    cumulative_totals_list = []
    all_rounds_data_list = []
    after_two_rounds_data_list = []  # Store data after 2 rounds for printing

    for simulation_num in range(1, num_simulations + 1):
        rounds_data = []
        players_not_making_cut = set()

        for round_num in range(1, 5):
            round_scores = pd.DataFrame(columns=['Player Name'] + [f'Hole {i+1}' for i in range(sum(holes.values()))])

            for player_name, player_data in df.groupby('Player Name'):
                if round_num > 2 and player_name in players_not_making_cut:
                    continue

                player_points = []
                for _, row in player_data.iterrows():
                    cluster = row['Cluster']
                    count = holes.get(cluster, 0)
                    probabilities = row[points].values.tolist()
                    probabilities /= np.sum(probabilities)
                    for _ in range(count):
                        player_points.append(simulate_round(probabilities, points))

                round_scores.loc[len(round_scores)] = [player_name] + player_points
                
            round_strokes = convert_points_to_strokes_and_accumulate(round_scores)
            round_scores['Total Points'] = round_scores.iloc[:, 1:-1].sum(axis=1)
            round_scores, _ = apply_bonus_points(round_scores)  # Apply consecutive three bonus
            round_scores = apply_bogey_free_round_bonus(round_scores)  # Apply Bogey Free Round bonus
            rounds_data.append(round_scores)

            if round_num == 2:
                round_strokes = convert_points_to_strokes_and_accumulate(round_scores)  # Apply the conversion function
                cumulative_total = round_strokes.groupby('Player Name')['Cumulative Strokes'].last().reset_index()
                cumulative_total_sorted = cumulative_total.sort_values(by='Cumulative Strokes', ascending=True)  # Sort by cumulative strokes
                players_to_cut = set(cumulative_total_sorted['Player Name'].tail(num_players_to_cut).values)
                players_not_making_cut.update(players_to_cut)

                
                after_two_rounds_data_list.append(round_scores)  # Store data after 2 rounds for printing

        all_rounds_data_list.append(rounds_data)
        cumulative_total = pd.concat(rounds_data).groupby('Player Name')['Total Points'].sum().reset_index()
        cumulative_total_sorted = cumulative_total.sort_values(by='Total Points', ascending=False)
        cumulative_totals_list.append(cumulative_total_sorted)

    return cumulative_totals_list, all_rounds_data_list, after_two_rounds_data_list

# Example usage:
num_players_to_cut = 20  # Define the number of players to cut
df_sims, all_rounds_data_list, after_two_rounds_data_list = simulate_multiple_4_rounds_with_cut_line(df, holes, points, num_players_to_cut, num_simulations=2)

# Print results after 2 rounds
print("Results after 2 rounds:")
for i, data in enumerate(after_two_rounds_data_list):
    print(f"Simulation {i+1}:")
    print(data)

'''


# In[ ]:





# In[ ]:





# In[ ]:





# In[355]:


#This is an attempt at cutting players by strokes and not fantasy points, 
#we are not quite there yet, I am unabe to integrate it the optimizer

'''

import pandas as pd
import numpy as np

def convert_points_to_strokes_and_accumulate(round_data):
    point_to_strokes = {-1: 2, -0.5: 1, 0.5: 0, 3: -1, 8: -2, 13: -3}
    strokes_round_data = round_data.copy()
    strokes_round_data['Cumulative Strokes'] = 0

    for index, row in strokes_round_data.iterrows():
        cumulative_strokes = 0
        for col in row.index:
            if col not in ['Player Name', 'Total Points', 'Cumulative Strokes']:
                strokes = point_to_strokes.get(row[col], 0)
                cumulative_strokes += strokes
                strokes_round_data.at[index, col] = cumulative_strokes
        strokes_round_data.at[index, 'Cumulative Strokes'] = cumulative_strokes

    return strokes_round_data

def simulate_multiple_4_rounds_with_cut_line(df, holes, points, num_players_to_cut, num_simulations=2):
    cumulative_totals_list = []
    all_rounds_data_list = []
    after_two_rounds_data_list = []

    for simulation_num in range(1, num_simulations + 1):
        rounds_data = []
        players_not_making_cut = set()

        for round_num in range(1, 5):
            round_scores = pd.DataFrame(columns=['Player Name'] + [f'Hole {i+1}' for i in range(sum(holes.values()))])

            for player_name, player_data in df.groupby('Player Name'):
                if round_num > 2 and player_name in players_not_making_cut:
                    continue

                player_points = []
                for _, row in player_data.iterrows():
                    cluster = row['Cluster']
                    count = holes.get(cluster, 0)
                    probabilities = row[points].values.tolist()
                    probabilities /= np.sum(probabilities)
                    for _ in range(count):
                        player_points.append(np.random.choice(points, p=probabilities))

                round_scores.loc[len(round_scores)] = [player_name] + player_points

            round_strokes = convert_points_to_strokes_and_accumulate(round_scores)
            round_scores['Total Points'] = round_scores.iloc[:, 1:-1].sum(axis=1)
            rounds_data.append(round_scores)

            if round_num == 2:
                round_strokes = convert_points_to_strokes_and_accumulate(round_scores)
                cumulative_total = round_strokes.groupby('Player Name')['Cumulative Strokes'].sum().reset_index()
                # Sort by cumulative strokes in ascending order
                cumulative_total_sorted = cumulative_total.sort_values(by='Cumulative Strokes', ascending=True)
                # Select the players with the highest (worst) scores to cut
                players_to_cut = set(cumulative_total_sorted['Player Name'].tail(num_players_to_cut).values)
                players_not_making_cut.update(players_to_cut)

                after_two_rounds_data_list.append(round_strokes)

                # Output the players being cut and their strokes
                print(f"\nPlayers being cut after simulation {simulation_num}:")
                for player in players_to_cut:
                    strokes = cumulative_total_sorted.loc[cumulative_total_sorted['Player Name'] == player, 'Cumulative Strokes'].values[0]
                    print(f"{player}: {strokes} strokes")

        all_rounds_data_list.append(rounds_data)
        cumulative_total = pd.concat(rounds_data).groupby('Player Name')['Total Points'].sum().reset_index()
        cumulative_totals_list.append(cumulative_total)

    return cumulative_totals_list, all_rounds_data_list, after_two_rounds_data_list


# Example usage (This requires you to have the 'df', 'holes', and 'points' variables defined as per your dataset)
num_players_to_cut = 80  # Define the number of players to cut
cumulative_totals, all_rounds_data, after_two_rounds_data = simulate_multiple_4_rounds_with_cut_line(df, holes, points, num_players_to_cut, num_simulations=2)

'''


# In[356]:


#after_two_rounds_data


# In[ ]:





# In[357]:


import pandas as pd

# Assuming 'original_data' is your nested list structure containing all dataframes
original_data = df_sims  # This should be replaced with your actual nested list structure

# Initialize two lists to hold the separated dataframes
summarized_points_data = []
detailed_hole_by_hole_data = []

# Define a function to handle the separation recursively
def separate_data(element):
    if isinstance(element, list):
        # If the element is a list, iterate over each item in the list
        for item in element:
            separate_data(item)
    elif isinstance(element, pd.DataFrame):
        # Check if the dataframe is of the summarized type or detailed type
        if 'Player Name' in element.columns and 'Total Points' in element.columns and len(element.columns) == 2:
            summarized_points_data.append(element)
        else:
            detailed_hole_by_hole_data.append(element)

# Start the separation process
for item in original_data:
    separate_data(item)

# At this point, 'summarized_points_data' will have all the summary dataframes,
# and 'detailed_hole_by_hole_data' will have all the detailed dataframes.


# In[358]:


detailed_hole_by_hole_data


# In[359]:


summarized_points_data


# In[360]:


# Filter out any item that is not a DataFrame
detailed_hole_by_hole_data = [item for item in detailed_hole_by_hole_data if isinstance(item, pd.DataFrame)]

# Now apply the function to each DataFrame
transformed_data_frames = [convert_points_to_strokes_and_accumulate(df) for df in detailed_hole_by_hole_data]


# In[361]:


import pandas as pd

# Ensure that the list contains only DataFrames
transformed_data_frames = []
for df in detailed_hole_by_hole_data:
    if isinstance(df, pd.DataFrame):
        transformed_df = convert_points_to_strokes_and_accumulate(df)
        transformed_data_frames.append(transformed_df)
    else:
        print("Item is not a DataFrame:", df)


# In[362]:


import pandas as pd

# Initialize an empty list to store the transformed DataFrames
transformed_data_frames = []

# Iterate over each item in the list
for df in detailed_hole_by_hole_data:
    # Check if the item is an Ellipsis and skip processing if it is
    if df is Ellipsis:
        print("Skipping Ellipsis...")
        continue
    
    # Check if the item is a DataFrame before processing
    if isinstance(df, pd.DataFrame):
        transformed_df = convert_points_to_strokes_and_accumulate(df)
        transformed_data_frames.append(transformed_df)
    else:
        print("Item is not a DataFrame and not Ellipsis:", type(df))


# In[363]:


import pandas as pd

# Define the function to convert points to strokes and accumulate them
def convert_points_to_strokes_and_accumulate(round_data):
    point_to_strokes = {-1: 2, -0.5: 1, 0.5: 0, 3: -1, 8: -2, 13: -3}
    strokes_round_data = round_data.copy()
    strokes_round_data['Cumulative Strokes'] = 0

    for index, row in strokes_round_data.iterrows():
        cumulative_strokes = 0
        for col in row.index:
            if col not in ['Player Name', 'Total Points', 'Cumulative Strokes']:
                strokes = point_to_strokes.get(row[col], 0)  # Default to 0 if not found
                cumulative_strokes += strokes
                strokes_round_data.at[index, col] = strokes
        strokes_round_data.at[index, 'Cumulative Strokes'] = cumulative_strokes

    return strokes_round_data

# Assuming detailed_hole_by_hole_data is your list of DataFrames
transformed_data_frames = [convert_points_to_strokes_and_accumulate(df) for df in detailed_hole_by_hole_data]

# Now, transformed_data_frames contains the modified DataFrames with cumulative strokes calculated.


# In[364]:


transformed_data_frames


# In[365]:


# Extracting every 4th DataFrame, assuming the first round you want starts at index 3 (4th position)
final_rounds = transformed_data_frames[3::4]
final_rounds


# In[366]:


for df in final_rounds:
    # Sort by 'Cumulative Strokes' in ascending order
    df.sort_values(by='Cumulative Strokes', inplace=True)
    
    # Assign initial ranks
    df['Rank'] = df['Cumulative Strokes'].rank(method='min')

    # Identify and process ties, but only for 1st place
    ties = df[df.duplicated(['Cumulative Strokes'], keep=False)]  # Get rows with ties
    tie_groups = ties.groupby('Cumulative Strokes')  # Group by the stroke counts

    for _, group in tie_groups:
        if len(group) > 1:  # Check if there are actual ties in the group
            min_rank = int(min(df.loc[group.index, 'Rank']))  # Get the minimum rank in the group
            if min_rank == 1:  # Only shuffle if the tie is for the 1st place
                shuffled_indices = np.random.permutation(group.index)
                for new_rank, idx in enumerate(shuffled_indices, start=min_rank):
                    df.at[idx, 'Rank'] = new_rank  # Assign new rank

    # Reset index if necessary
    df.reset_index(drop=True, inplace=True)


# In[367]:


#final_rounds


# In[368]:


# Define the points for each position
position_points = {
    1: 30, 2: 20, 3: 18, 4: 16, 5: 14, 6: 12, 7: 10, 8: 9, 9: 8, 10: 7,
    11: 6, 12: 6, 13: 6, 14: 6, 15: 6, 16: 5, 17: 5, 18: 5, 19: 5, 20: 5,
    21: 4, 22: 4, 23: 4, 24: 4, 25: 4, 26: 3, 27: 3, 28: 3, 29: 3, 30: 3,
    31: 2, 32: 2, 33: 2, 34: 2, 35: 2, 36: 2, 37: 2, 38: 2, 39: 2, 40: 2,
    41: 1, 42: 1, 43: 1, 44: 1, 45: 1, 46: 1, 47: 1, 48: 1, 49: 1, 50: 1
}

# Function to add position points based on the rank
def add_position_points(df):
    # Mapping the Rank to Position Points
    df['Position Points'] = df['Rank'].map(position_points).fillna(0)  # Fill NaN with 0 if rank is beyond 50
    return df

# Applying the function to each DataFrame in the list
final_round_dfs_with_points = [add_position_points(df) for df in final_rounds]
final_round_dfs_with_points


# In[ ]:





# In[369]:


updated_summarized_data = []

for summarized_df, final_round_df in zip(summarized_points_data, final_round_dfs_with_points):
    # Merge the dataframes on the Player Name
    merged_df = pd.merge(summarized_df, final_round_df[['Player Name', 'Position Points']],
                         on='Player Name', how='left')

    # If a player in summarized_points_data does not exist in final_round_dfs_with_points, fill with 0
    merged_df['Position Points'] = merged_df['Position Points'].fillna(0)

    # Add the Position Points to the Total Points
    merged_df['Total Points'] = merged_df['Total Points'] + merged_df['Position Points']

    # Keep the relevant columns
    updated_df = merged_df[['Player Name', 'Total Points']]

    # Append the updated DataFrame to the new list
    updated_summarized_data.append(updated_df)

# Now, updated_summarized_data contains all the updated DataFrames


# In[370]:


simulation_df = updated_summarized_data
simulation_df


# In[ ]:





# In[371]:


from pydfs_lineup_optimizer import get_optimizer, Site, Sport, TeamStack, PlayerFilter
from pydfs_lineup_optimizer import PlayersGroup, PositionsStack, Stack, statistics
import os


# In[372]:


df_2 = pd.read_csv(r'/Users/danielbrown/Desktop/DKSalaries-8.csv')
df_2


# In[ ]:





# In[373]:


import os  # Import the os module

output_csv_folder = '/Users/danielbrown/Desktop/'
all_lineups = []  # List to store all generated lineups

# Assuming df_2 exists and is correctly structured
# simulation_dfs is your list of DataFrames

for index, df in enumerate(simulation_df):
    # Rename columns to standardize
    df.rename(columns={'Player Name': 'Name', 'Total Points': 'AvgPointsPerGame'}, inplace=True)
    
    # Perform the merge
    merged_df = pd.merge(df_2, df, on='Name', how='left', suffixes=('_x', '_y'))

    # Use 'AvgPointsPerGame' from the simulation results, and drop any other 'AvgPointsPerGame' column
    merged_df['AvgPointsPerGame'] = merged_df['AvgPointsPerGame_y'].fillna(merged_df['AvgPointsPerGame_x'])
    merged_df.drop(columns=['AvgPointsPerGame_x', 'AvgPointsPerGame_y'], inplace=True)

    # Fill NaN values in 'AvgPointsPerGame' with zero and convert the column to float
    merged_df['AvgPointsPerGame'] = pd.to_numeric(merged_df['AvgPointsPerGame'], errors='coerce').fillna(0)

    # Filter out the rows where 'AvgPointsPerGame' is zero after filling NaNs
    merged_df = merged_df[merged_df['AvgPointsPerGame'] > 0]

    if merged_df.empty:
        print(f"No players with non-zero AvgPointsPerGame found at index {index}. Continuing to next simulation.")
        continue

    # Save the filtered DataFrame to a CSV file
    csv_path = os.path.join(output_csv_folder, f'merged_df_simulation_{index}_hole_by_hole.csv')
    merged_df.to_csv(csv_path, index=False)

    # Create the optimizer object and generate lineups
    # Note: Ensure the optimizer setup is correctly implemented as per your environment
    optimizer = get_optimizer(Site.DRAFTKINGS, Sport.GOLF)
    optimizer.load_players_from_csv(csv_path)
    lineups = optimizer.optimize(n=1)

    # Append generated lineups to all_lineups
    all_lineups.extend(lineups)

    # Delete the CSV file after it's used
    os.remove(csv_path)

# all_lineups now contains all the generated lineups for each simulation result


# In[374]:


# Printing the all_lineups list to verify the appended data
for lineup in all_lineups:
    print(lineup)


# In[375]:


lineup_data = []

# Iterate over all_lineups and extract lineup information
for lineup in all_lineups:
    lineup_info = [player.full_name for player in lineup.players]
    lineup_info.extend([lineup.salary_costs, lineup.fantasy_points_projection])
    lineup_data.append(lineup_info)

# Create a DataFrame from the lineup data
df = pd.DataFrame(lineup_data, columns=['G','G','G','G','G','G','Budget','FPPG'])

# Export the DataFrame to a CSV file
df.to_csv('/Users/danielbrown/Desktop/Golf_Sim_output_holebyhole.csv', index=False)


# In[376]:


'''
# Filter the DataFrame for 'Budget' > 45000
df = df[df['Budget'] > 49200]
# Check the length of the DataFrame df
df_length = len(df)



# Print the length of the DataFrame
print("Length of DataFrame df:", df_length)
df.to_csv('/Users/danielbrown/Desktop/Golf_Sim_output_12.csv', index=False)
df
'''


# In[377]:


from collections import defaultdict

# Create a defaultdict to store player counts
player_counts = defaultdict(int)

# Iterate over each row in the DataFrame
for index, row in df.iterrows():
    # Use a set to keep track of players seen in each lineup
    seen_players = set()
    # Iterate over the 'G' columns (assuming these columns contain player names)
    for player in row[['G', 'G', 'G', 'G', 'G', 'G']]:  # Assuming 6 'G' columns in your DataFrame
        # Check if the player has not been seen in this lineup
        if player not in seen_players:
            player_counts[player] += 1
            seen_players.add(player)

# Convert player counts to a DataFrame
player_counts_df = pd.DataFrame(list(player_counts.items()), columns=['Player', 'Count'])

# Sort DataFrame by player count in descending order
player_counts_df = player_counts_df.sort_values(by='Count', ascending=False)

# Export DataFrame to CSV
player_counts_df.to_csv('/Users/danielbrown/Desktop/player_counts.csv', index=False)
player_counts_df


# In[378]:


# Create a dictionary mapping player names to names + IDs
name_id_dict = df_2.set_index('Name')['Name + ID'].to_dict()

# Merge the dataframes on the names using the dictionary
df_merged = df.replace(name_id_dict)

# Reset the index of df_merged
df_merged.reset_index(drop=True, inplace=True)

print(df_merged)


# In[379]:


df_merged.to_csv('/Users/danielbrown/Desktop/golf_sim_holebyhole.csv', index=False)


# In[386]:


# Filter the DataFrame for 'Budget' > X
df_merged = df_merged[df_merged['Budget'] > 47000]
# Check the length of the DataFrame df
df_length = len(df_merged)



# Print the length of the DataFrame
print("Length of DataFrame df:", df_length)
df_merged.to_csv('/Users/danielbrown/Desktop/Golf_Sim_masters_2.csv', index=False)
df_merged
df = df_merged


# In[381]:


from collections import defaultdict

# Create a defaultdict to store player counts
player_counts = defaultdict(int)

# Iterate over each row in the DataFrame
for index, row in df.iterrows():
    # Use a set to keep track of players seen in each lineup
    seen_players = set()
    # Iterate over the 'G' columns (assuming these columns contain player names)
    for player in row[['G', 'G', 'G', 'G', 'G', 'G']]:  # Assuming 6 'G' columns in your DataFrame
        # Check if the player has not been seen in this lineup
        if player not in seen_players:
            player_counts[player] += 1
            seen_players.add(player)

# Convert player counts to a DataFrame
player_counts_df = pd.DataFrame(list(player_counts.items()), columns=['Player', 'Count'])

# Sort DataFrame by player count in descending order
player_counts_df = player_counts_df.sort_values(by='Count', ascending=False)

# Export DataFrame to CSV
player_counts_df.to_csv('/Users/danielbrown/Desktop/player_counts.csv', index=False)
player_counts_df


# In[382]:


'''

import pandas as pd
import random

def sample_fixed_rows(df, num_rows):
    """
    Samples a fixed number of rows from the dataframe.

    Parameters:
    df (pd.DataFrame): The dataframe to sample from.
    num_rows (int): The exact number of rows to sample. If num_rows is greater than the 
                    number of rows in the dataframe, it will sample all rows.

    Returns:
    pd.DataFrame: A dataframe containing the randomly sampled rows.
    """
    if num_rows > len(df):
        # If num_rows exceeds the number of rows in the dataframe, sample all rows
        return df
    else:
        # Randomly sample the specified number of rows
        return df.sample(n=num_rows)

# Example usage:
# Assuming df_merged is your dataframe and you want to sample exactly 10 rows
random_sample = sample_fixed_rows(df_merged, )
print(random_sample)


random_sample.to_csv('/Users/danielbrown/Desktop/golf_sim_lineups.csv', index=False)

'''


# In[ ]:





# In[383]:


import pandas as pd
import re
from itertools import combinations
from collections import Counter

# Load the CSV file
df = pd.read_csv('/Users/danielbrown/Desktop/Golf_Sim_output_1.csv')

# Remove the last two columns
df = df.iloc[:, :-2]

# Function to clean the cell by removing numbers and parentheses
def clean_name(name):
    # Regular expression to find numbers and parentheses and replace them with an empty string
    return re.sub(r'\(\d+\)', '', name).strip()

# Apply the cleaning function to each cell in the DataFrame
for col in df.columns:
    df[col] = df[col].apply(clean_name)

# Now df is preprocessed, you can proceed with the counting logic as described earlier
print(df)


# In[384]:


# Load the CSV file
#df = pd.read_csv('/Users/danielbrown/Desktop/Draftkings_Golf_Round1_Names.csv')

# Initialize counters
pair_counts = Counter()
triplet_counts = Counter()

# Iterate through each lineup
for index, row in df.iterrows():
    lineup = row.tolist()
    
    # Count pairs
    for pair in combinations(lineup, 2):
        pair_counts[pair] += 1
    
    # Count triplets
    for triplet in combinations(lineup, 3):
        triplet_counts[triplet] += 1

# Display pair counts
print("Pair counts:")
for pair, count in pair_counts.items():
    print(f"{pair}: {count}")

# Display triplet counts
print("\nTriplet counts:")
for triplet, count in triplet_counts.items():
    print(f"{triplet}: {count}")


# In[385]:


# Function to sort and filter counts
def sort_and_filter_counts(counts, threshold):
    # Sort the items by count in descending order
    sorted_counts = sorted(counts.items(), key=lambda x: x[1], reverse=True)
    # Filter out counts below the threshold
    filtered_counts = [item for item in sorted_counts if item[1] >= threshold]
    return filtered_counts

# Define your threshold here
threshold = 3  # Example threshold

# Sort and filter pairs and triplets
sorted_filtered_pairs = sort_and_filter_counts(pair_counts, threshold)
sorted_filtered_triplets = sort_and_filter_counts(triplet_counts, threshold)

# Display the sorted and filtered pair counts
print("Sorted and filtered pair counts:")
for pair, count in sorted_filtered_pairs:
    print(f"{pair}: {count}")

# Display the sorted and filtered triplet counts
print("\nSorted and filtered triplet counts:")
for triplet, count in sorted_filtered_triplets:
    print(f"{triplet}: {count}")


# In[ ]:




