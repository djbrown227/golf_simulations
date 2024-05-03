#!/usr/bin/env python
# coding: utf-8

# In[21]:


import pandas as pd
import numpy as np
from pydfs_lineup_optimizer import get_optimizer, Site, Sport, TeamStack, PlayerFilter
from pydfs_lineup_optimizer import PlayersGroup, PositionsStack, Stack, statistics
from scipy.stats import shapiro
import matplotlib.pyplot as plt


# In[22]:


df = pd.read_csv('/Users/danielbrown/Desktop/Golfer_Dataframe_total.csv')
df_prob = pd.read_excel('/Users/danielbrown/Desktop/rbcheritage.xlsx')
# Remove rows with NaN values
df_prob = df_prob.dropna()
df_prob


# In[23]:


df_prob = df_prob[['Name','Prob Top 10','Prob Make Cut']]
# Converting percentage columns to decimals
#df_prob['Win%'] = df_prob['Win%'].str.rstrip('%').astype('float') / 100
df_prob['Prob Top 10'] = df_prob['Prob Top 10'].str.rstrip('%').astype('float') / 100
df_prob['Prob Make Cut'] = df_prob['Prob Make Cut'].str.rstrip('%').astype('float') / 100
df_prob


# In[24]:


# Convert 'Event Date' column to datetime format
df['Event Date'] = pd.to_datetime(df['Event Date'])

# Extract year from 'Event Date' column
df['Year'] = df['Event Date'].dt.year
df


# In[25]:


# Group by Event and Year and get the original event date for each group
original_dates = df[df['Round'] == 1].groupby(['Event Title', 'Year'])['Event Date'].first().reset_index()

# Merge original dates back to the DataFrame
df = pd.merge(df, original_dates, on=['Event Title', 'Year'], suffixes=('', '_original'))

# Calculate modified dates
df['Modified Date'] = df['Event Date_original'] + pd.to_timedelta(df['Round'] - 1, unit='D')

# Filter out rows where Cumulative_FPTS is greater than 200
#df = df[df['Cumulative_FPTS'] <= 200]

# Display the DataFrame with the new 'Modified Date' column
#print(df[['Event', 'Year', 'Round', 'Modified Date']])
df


# In[26]:


import pandas as pd

# Assuming your dataframe is named df

# Filter for Rd 4 Hole 18 rows where Score does not equal "CUT"
rd_4_not_cut = df[(df['Round'] == 4) & (df['Hole'] == 18) & (df['Score'] != 'CUT')]
rd_4_not_cut = rd_4_not_cut[~rd_4_not_cut['Player Name'].str.contains('/')]

# Filter for Rd 2 Hole 18 rows where Score equals "CUT"
rd_2_cut = df[(df['Round'] == 2) & (df['Hole'] == 18) & (df['Score'] == 'CUT')]
# Filter out rows with "/" in the column Player Name
rd_2_cut = rd_2_cut[~rd_2_cut['Player Name'].str.contains('/')]

# Create a subset where 'Position_At_Hole' is equal to or less than 10
rd_4_top10 = rd_4_not_cut[rd_4_not_cut['Position_At_Hole'] <= 10]

# Remove rows that are in rd_4_top10 from rd_4_not_cut
rd_4_not_cut = rd_4_not_cut[~rd_4_not_cut.index.isin(rd_4_top10.index)]

# Display the results
print("Rd 4 Hole 18 rows where Score does not equal 'CUT':")
print(rd_4_not_cut)

print("\nRd 2 Hole 18 rows where Score equals 'CUT':")
print(rd_2_cut)

# Save the DataFrames to CSV files
rd_4_not_cut.to_csv(r"/Users/danielbrown/Desktop/rd_4_not_cut.csv")
rd_2_cut.to_csv(r"/Users/danielbrown/Desktop/rd_2_cut.csv")
rd_4_top10.to_csv(r"/Users/danielbrown/Desktop/rd_4_top10.csv")


# In[27]:


rd_2_cut


# In[28]:


# Filter the dataframe to include only rows where 'Cumulative_FPTS' is between 0 and 40
rd_2_cut_distr = rd_2_cut[(rd_2_cut['Cumulative_FPTS'] >= 0) & (rd_2_cut['Cumulative_FPTS'] <= 50)]
rd_2_cut_distr = rd_2_cut_distr[['Player Name','Cumulative_FPTS']]

rd_2_cut_distr


# In[29]:


import pandas as pd

# Assuming rd_2_cut_distr is your dataframe

# Define the bins for the ranges
bins = [0, 10, 20, 30, 40, 50]

# Define the labels for the ranges
labels = ['0-10', '10-20', '20-30', '30-40', '40-50']

# Bin the 'Cumulative_FPTS' column into the specified ranges
rd_2_cut_distr['FPTS_Range'] = pd.cut(rd_2_cut_distr['Cumulative_FPTS'], bins=bins, labels=labels, right=False)

# Group by 'Player Name' and 'FPTS_Range', then count occurrences and normalize
grouped = rd_2_cut_distr.groupby(['Player Name', 'FPTS_Range']).size().unstack(fill_value=0)
probability_distribution = grouped.div(grouped.sum(axis=1), axis=0)

# Reorder the columns
probability_distribution = probability_distribution[['0-10', '10-20', '20-30', '30-40', '40-50']]

# Calculate the total number of data points for each player
data_points_count = grouped.sum(axis=1)

# Add the data points count as a new column to the probability distribution dataframe
probability_distribution['Data Points'] = data_points_count

# Sort the DataFrame by the '40-50' column in descending order
sorted_df_cut = probability_distribution.sort_values(by='40-50', ascending=False)
sorted_df_cut = sorted_df_cut.dropna()
# Save the sorted DataFrame to a CSV file
#sorted_df.to_csv('probability_distribution_with_counts.csv')

print(sorted_df_cut)


# In[30]:


#We may want to add in if Data Points < 5 then to have a uniform disribution across all bins

list(sorted_df_cut.columns)


# In[31]:


'''

#Test if Top Ten is Normal
statistic, p_value = shapiro(rd_2_cut_distr)

print('Shapiro-Wilk Test Statistic:', statistic)
print('p-value:', p_value)

if p_value > 0.05:
    print("The distribution appears to be normal.")
else:
    print("The distribution does not appear to be normal.")
    
'''


# In[32]:


'''

# Plotting the histogram
plt.figure(figsize=(8, 6))
plt.hist(rd_2_cut_distr, bins=100, edgecolor='black')
plt.title('Distribution of FPTS for Players NOT Making the Cut')
plt.xlabel('FPTS')
plt.ylabel('Frequency')
plt.grid(axis='y', alpha=0.75)
plt.show()

'''


# In[33]:


# Sort the DataFrame by the '40-50' column in descending order
rd_4_not_cut = rd_4_not_cut.sort_values(by='Cumulative_FPTS', ascending=False)
rd_4_not_cut
rd_4_not_cut.to_csv(r"/Users/danielbrown/Desktop/rd_4_not_cut.csv")


# In[34]:


rd_4_not_cut = rd_4_not_cut[['Player Name','Cumulative_FPTS']]
rd_4_not_cut


# In[35]:


import pandas as pd

# Create the bins and labels for the specified range
bins = list(range(30, 130, 10))  # 30, 40, ..., 120
labels = [f'{i}-{i+10}' for i in bins[:-1]]  # '30-40', '40-50', ..., '110-120'

# Assuming rd_4_not_cut is your dataframe
# Bin the 'Cumulative_FPTS' column
rd_4_not_cut['FPTS_Range'] = pd.cut(rd_4_not_cut['Cumulative_FPTS'], bins=bins, labels=labels, right=False)

# Group by 'Player Name' and 'FPTS_Range', count occurrences, and normalize
grouped = rd_4_not_cut.groupby(['Player Name', 'FPTS_Range']).size().unstack(fill_value=0)
probability_distribution = grouped.div(grouped.sum(axis=1), axis=0)

# Reorder the columns based on the labels
probability_distribution = probability_distribution[labels]

# Calculate the total number of data points for each player
data_points_count = grouped.sum(axis=1)

# Add the data points count as a new column to the probability distribution dataframe
probability_distribution['Data Points'] = data_points_count

# Sort the DataFrame by a specific column in descending order (you can change the column as needed)
sorted_df_notcut = probability_distribution.sort_values(by='110-120', ascending=False)

sorted_df_notcut = sorted_df_notcut.dropna()
# Uncomment the following line to save the sorted DataFrame to a CSV file
# sorted_df.to_csv('probability_distribution_with_counts_rd_4.csv')

# Display the sorted DataFrame
print(sorted_df_notcut)


# In[36]:


#descriptive_stats = sorted_df.describe(include='all')
#descriptive_stats


# In[37]:


#Checking for normality
#Isolate Top Ten Data
rd_4_top10 = rd_4_top10[['Player Name','Cumulative_FPTS']]
#rd_4_not_cut_distr
# Get descriptive statistics
descriptive_stats = rd_4_top10.describe()

print(descriptive_stats)


# In[38]:


import pandas as pd

# Create the bins and labels for the specified range
bins = list(range(60, 150, 10))  # 60, 70, ..., 140
labels = [f'{i}-{i+10}' for i in bins[:-1]]  # '60-70', '70-80', ..., '130-140'

# Assuming rd_4_top10 is your dataframe
# Remove rows with any NaN values
rd_4_top10_clean = rd_4_top10.dropna()

# Bin the 'Cumulative_FPTS' column
rd_4_top10_clean['FPTS_Range'] = pd.cut(rd_4_top10_clean['Cumulative_FPTS'], bins=bins, labels=labels, right=False)

# Group by 'Player Name' and 'FPTS_Range', count occurrences, and normalize
grouped = rd_4_top10_clean.groupby(['Player Name', 'FPTS_Range']).size().unstack(fill_value=0)
probability_distribution = grouped.div(grouped.sum(axis=1), axis=0)

# Reorder the columns based on the labels
probability_distribution = probability_distribution[labels]

# Calculate the total number of data points for each player
data_points_count = grouped.sum(axis=1)

# Add the data points count as a new column to the probability distribution dataframe
probability_distribution['Data Points'] = data_points_count

# Sort the DataFrame by a specific column in descending order (you can change the column as needed)
sorted_df_top10 = probability_distribution.sort_values(by='130-140', ascending=False)
sorted_df_top10 = sorted_df_top10.dropna()
# Uncomment the following line to save the sorted DataFrame to a CSV file
# sorted_df.to_csv('probability_distribution_with_counts_rd_4_top10_clean.csv')

# Display the sorted DataFrame
print(sorted_df_top10)


# In[ ]:





# In[39]:


'''

import pandas as pd
import numpy as np

# Assuming rd_4_not_cut and rd_2_cut are the dataframes filtered earlier

import numpy as np

# Assuming rd_4_not_cut and rd_2_cut are the dataframes filtered earlier

# Calculate mean, variance, and standard deviation of Cumulative_FPTS for Rd 4 Hole 18 rows
rd_4_stats = rd_4_not_cut.groupby('Player Name')['Cumulative_FPTS'].agg(['mean', 'var', 'count']).reset_index()
rd_4_stats['std'] = rd_4_stats['var'].apply(np.sqrt)
rd_4_stats.columns = ['Player Name', 'Mean_Rd4', 'Variance_Rd4', 'Count_Rd4', 'Std_Rd4']

# Calculate mean, variance, and standard deviation of Cumulative_FPTS for Rd 2 Hole 18 rows
rd_2_stats = rd_2_cut.groupby('Player Name')['Cumulative_FPTS'].agg(['mean', 'var', 'count']).reset_index()
rd_2_stats['std'] = rd_2_stats['var'].apply(np.sqrt)
rd_2_stats.columns = ['Player Name', 'Mean_Rd2', 'Variance_Rd2', 'Count_Rd2', 'Std_Rd2']


# Merge the dataframes on player name
combined_stats = pd.merge(rd_4_stats, rd_2_stats, on='Player Name', how='outer')

# Fill NaN values with 0 (if a player only appeared in one dataframe)
combined_stats.fillna(0, inplace=True)

# Sort by player name
combined_stats.sort_values(by='Mean_Rd4', inplace=True, ascending=False)

#Only include guys who played more than or equal to 32 rounds of golf in this time period
combined_stats = combined_stats[combined_stats['Count_Rd4'] >= 8]

# Reset the index
combined_stats = combined_stats.reset_index(drop=True)

# Display the combined and sorted dataframe
print("Combined and Sorted Stats:")
#print(combined_stats)
combined_stats.head(50)
combined_stats

'''


# In[40]:


'''

# Calculate the average of Mean_Rd2 and Std_Rd2 excluding rows where Count_Rd2 is 0
mean_rd2_avg = combined_stats.loc[combined_stats['Count_Rd2'] != 0, 'Mean_Rd2'].mean()
std_rd2_avg = combined_stats.loc[combined_stats['Count_Rd2'] != 0, 'Std_Rd2'].mean()

# Replace 0s in Mean_Rd2 and Std_Rd2 with their respective averages for rows where Count_Rd2 is 0
combined_stats.loc[combined_stats['Count_Rd2'] == 0, 'Mean_Rd2'] = mean_rd2_avg
combined_stats.loc[combined_stats['Count_Rd2'] == 0, 'Std_Rd2'] = std_rd2_avg

# Now combined_stats is updated, and you can work with it further or display it
combined_stats = combined_stats[['Player Name','Mean_Rd4','Std_Rd4','Mean_Rd2','Std_Rd2']]
print(combined_stats)

'''


# In[41]:


#combined_stats.to_csv(r"/Users/danielbrown/Desktop/player_distributions_individual.csv")


# In[42]:


'''

import numpy as np
import pandas as pd

# Assuming df_prob and combined_stats are already defined

# Merging the dataframes on golfer names to align the probabilities with their stats
simulation_df = pd.merge(df_prob, combined_stats, left_on='Name', right_on='Player Name')

# Defining the simulation function
def simulate_golfer_performance(golfer):
    golfer = golfer.copy()
    make_cut = np.random.rand() < golfer['Prob Make Cut']
    golfer['Made Cut'] = 1 if make_cut else 0
    
    if make_cut:
        # Golfer makes the cut, use Mean_Rd4 and Std_Rd4
        points = np.random.normal(golfer['Mean_Rd4'], golfer['Std_Rd4'])
        # Calculate the conditional probability of being in the top 10, given the golfer makes the cut
        prob_top_10_given_cut = golfer['Prob Top 10'] / golfer['Prob Make Cut']
        in_top_10 = np.random.rand() < prob_top_10_given_cut
        golfer['Top 10'] = 1 if in_top_10 else 0
    else:
        # Golfer does not make the cut, use Mean_Rd2 and Std_Rd2
        points = np.random.normal(golfer['Mean_Rd2'], golfer['Std_Rd2'])
        golfer['Top 10'] = 0

    golfer['AvgPointsPerGame'] = max(points, 0)  # Ensure non-negative points
    return golfer

# Function to apply the simulation across all golfers
def simulate_round(df):
    simulated_data = df.apply(simulate_golfer_performance, axis=1)
    return simulated_data

# Simulate the round
simulated_round = simulate_round(simulation_df)
simulated_round.sort_values(by='AvgPointsPerGame', inplace=True, ascending=False)
simulated_round

'''


# In[ ]:





# In[43]:


'''

import pandas as pd
import numpy as np

# Assuming df_prob, sorted_df_cut, sorted_df_notcut, and sorted_df_top10 are already defined

num_simulations = 100
simulations_results = []

def simulate_one_round(df_prob, sorted_df_cut, sorted_df_notcut, sorted_df_top10):
    results = {}
    for index, row in df_prob.iterrows():
        name = row['Name']  # This corresponds to the 'Name' column in df_prob
        make_cut = np.random.choice([True, False], p=[row['Prob Make Cut'], 1 - row['Prob Make Cut']])
        
        if make_cut:
            top_10 = np.random.choice([True, False], p=[row['Prob Top 10'], 1 - row['Prob Top 10']])
            if top_10 and name in sorted_df_top10.index:
                prob_dist = sorted_df_top10.loc[name]
            elif not top_10 and name in sorted_df_notcut.index:
                prob_dist = sorted_df_notcut.loc[name]
            else:
                prob_dist = pd.Series()
        else:
            if name in sorted_df_cut.index:
                prob_dist = sorted_df_cut.loc[name]
            else:
                prob_dist = pd.Series()
        
        # Drop 'Data Points' if it exists
        if 'Data Points' in prob_dist:
            prob_dist = prob_dist.drop('Data Points')
        
        # Normalize and choose a score
        if not prob_dist.empty:
            probabilities = prob_dist.values / prob_dist.values.sum()
            score_bin = np.random.choice(prob_dist.index, p=probabilities)
            results[name] = np.mean([int(s) for s in score_bin.split('-')])
        else:
            results[name] = 0

    return pd.DataFrame.from_dict(results, orient='index', columns=['Score'])

# Run the simulations
for _ in range(num_simulations):
    simulation_result = simulate_one_round(df_prob, sorted_df_cut, sorted_df_notcut, sorted_df_top10)
    simulations_results.append(simulation_result)

# simulations_results contains 1000 DataFrames, each representing one simulation of the tournament

'''


# In[44]:


#simulations_results


# In[ ]:





# In[45]:


import pandas as pd
import numpy as np

# Assuming df_prob, sorted_df_cut, sorted_df_notcut, and sorted_df_top10 are already defined

num_simulations = 1000
simulations_results = []

def simulate_one_round(df_prob, sorted_df_cut, sorted_df_notcut, sorted_df_top10):
    results = {}
    for index, row in df_prob.iterrows():
        name = row['Name']  # This corresponds to the 'Name' column in df_prob
        make_cut = np.random.choice([True, False], p=[row['Prob Make Cut'], 1 - row['Prob Make Cut']])
        
        if make_cut:
            top_10 = np.random.choice([True, False], p=[row['Prob Top 10'], 1 - row['Prob Top 10']])
            if top_10 and name in sorted_df_top10.index:
                prob_dist = sorted_df_top10.loc[name]
            elif not top_10 and name in sorted_df_notcut.index:
                prob_dist = sorted_df_notcut.loc[name]
            else:
                prob_dist = pd.Series()
        else:
            if name in sorted_df_cut.index:
                prob_dist = sorted_df_cut.loc[name]
            else:
                prob_dist = pd.Series()

        # Drop 'Data Points' if it exists
        if 'Data Points' in prob_dist:
            prob_dist = prob_dist.drop('Data Points')

        # Normalize and choose a score bin
        if not prob_dist.empty:
            probabilities = prob_dist.values / prob_dist.values.sum()
            selected_bin = np.random.choice(prob_dist.index, p=probabilities)
            bin_min, bin_max = [int(x) for x in selected_bin.split('-')]
            # Uniformly select a score within the chosen bin range
            results[name] = np.random.randint(bin_min, bin_max+1)
        else:
            results[name] = 0

    return pd.DataFrame.from_dict(results, orient='index', columns=['Score'])

# Run the simulations
for _ in range(num_simulations):
    simulation_result = simulate_one_round(df_prob, sorted_df_cut, sorted_df_notcut, sorted_df_top10)
    simulations_results.append(simulation_result)

# simulations_results contains 1000 DataFrames, each representing one simulation of the tournament


# In[46]:


simulations_results


# In[47]:


# Iterate over each simulation result and adjust the column names
adjusted_simulations_results = []

for df in simulations_results:
    df = df.reset_index().rename(columns={'index': 'Name', 'Score': 'AvgPointsPerGame'})
    adjusted_simulations_results.append(df)

# Now, adjusted_simulations_results contains DataFrames with 'Name' and 'AvgPointsPerGame' columns
adjusted_simulations_results


# In[48]:


from pydfs_lineup_optimizer import get_optimizer, Site, Sport, TeamStack, PlayerFilter
from pydfs_lineup_optimizer import PlayersGroup, PositionsStack, Stack, statistics
import os


# In[49]:


df_2 = pd.read_csv(r'/Users/danielbrown/Desktop/DKSalaries-8 - DKSalaries-8.csv')
df_2


# In[50]:


'''

output_csv_folder = '/Users/danielbrown/Desktop/'
all_lineups = []  # List to store all generated lineups

# Assuming df_2 is your DataFrame from the CSV and adjusted_simulations_results is your list of simulation results
for index, simulation_df in enumerate(adjusted_simulations_results):
    # Merge the simulation results with the salaries and other details from df_2
    # Ensuring the 'Name' column matches between the simulation DataFrame and df_2
    merged_df = pd.merge(df_2, simulation_df, on='Name', how='left')

    # Use 'AvgPointsPerGame' from the simulation results, and drop the original 'AvgPointsPerGame' column if needed
    # Assuming that 'AvgPointsPerGame_x' comes from df_2 and 'AvgPointsPerGame_y' comes from the simulation
    merged_df['AvgPointsPerGame'] = merged_df['AvgPointsPerGame_y'].fillna(merged_df['AvgPointsPerGame_x'])
    merged_df.drop(columns=['AvgPointsPerGame_x', 'AvgPointsPerGame_y'], inplace=True)

    # Ensure 'AvgPointsPerGame' is numeric and handle NaNs
    merged_df['AvgPointsPerGame'] = pd.to_numeric(merged_df['AvgPointsPerGame'], errors='coerce').fillna(0)

    # Save the merged DataFrame to a CSV file for optimization
    csv_path = os.path.join(output_csv_folder, f'merged_df_simulation_{index}.csv')
    merged_df.to_csv(csv_path, index=False)

    # Create optimizer object and generate lineups
    optimizer = get_optimizer(Site.DRAFTKINGS, Sport.GOLF)
    optimizer.load_players_from_csv(csv_path)
    lineups = optimizer.optimize(n=1)  # Generate one lineup for the example; change 'n' as needed

    # Append generated lineups to all_lineups
    all_lineups.extend(lineups)

    # Clean up by deleting the CSV file after use
    os.remove(csv_path)

# Now, all_lineups contains the generated lineups for each simulation result
'''


# In[51]:


import os
from pydfs_lineup_optimizer import get_optimizer, Site, Sport

output_csv_folder = '/Users/danielbrown/Desktop/'
all_lineups = []  # List to store all generated lineups

# Assuming df_2 is your DataFrame from the CSV
for index, simulation_df in enumerate(adjusted_simulations_results):
    # Merge the original df_2 with the simulation DataFrame
    # The simulation DataFrame should have 'Name' and 'AvgPointsPerGame' columns
    merged_df = pd.merge(df_2.drop(columns=['AvgPointsPerGame']), simulation_df, on='Name', how='left', suffixes=('_x', '_y'))

    # After merging, rename 'AvgPointsPerGame_y' to 'AvgPointsPerGame' to use the simulation data
    # If 'AvgPointsPerGame_y' does not exist, it means none of the players were in the simulation
    if 'AvgPointsPerGame_y' in merged_df.columns:
        merged_df.rename(columns={'AvgPointsPerGame_y': 'AvgPointsPerGame'}, inplace=True)

    # If the simulation data was missing for some players, you might want to handle it (e.g., set to 0 or keep original)
    merged_df['AvgPointsPerGame'].fillna(0, inplace=True)

    # Now, ensure there's no 'AvgPointsPerGame_x' left in the DataFrame
    merged_df.drop(columns=['AvgPointsPerGame_x'], inplace=True, errors='ignore')

    # Save the merged DataFrame to a CSV file for optimization
    csv_path = os.path.join(output_csv_folder, f'merged_df_simulation_{index}.csv')
    merged_df.to_csv(csv_path, index=False)

    # Create optimizer object and generate lineups
    optimizer = get_optimizer(Site.DRAFTKINGS, Sport.GOLF)
    optimizer.load_players_from_csv(csv_path)
    lineups = optimizer.optimize(n=1)  # Generate one lineup for the example; change 'n' as needed

    # Append generated lineups to all_lineups
    for lineup in lineups:
        all_lineups.append(lineup)

    # Clean up by deleting the CSV file after use
    os.remove(csv_path)

# Now, all_lineups contains the generated lineups for each simulation result


# In[52]:


# Printing the all_lineups list to verify the appended data
for lineup in all_lineups:
    print(lineup)


# In[53]:


lineup_data = []

# Iterate over all_lineups and extract lineup information
for lineup in all_lineups:
    lineup_info = [player.full_name for player in lineup.players]
    lineup_info.extend([lineup.salary_costs, lineup.fantasy_points_projection])
    lineup_data.append(lineup_info)

# Create a DataFrame from the lineup data
df = pd.DataFrame(lineup_data, columns=['G','G','G','G','G','G','Budget','FPPG'])

# Export the DataFrame to a CSV file
#df.to_csv('/Users/danielbrown/Desktop/Golf_Sim_output_holebyhole.csv', index=False)


# In[54]:


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
#player_counts_df.to_csv('/Users/danielbrown/Desktop/player_counts.csv', index=False)
player_counts_df


# In[55]:


# Create a dictionary mapping player names to names + IDs
name_id_dict = df_2.set_index('Name')['Name + ID'].to_dict()

# Merge the dataframes on the names using the dictionary
df_merged = df.replace(name_id_dict)

# Reset the index of df_merged
df_merged.reset_index(drop=True, inplace=True)

print(df_merged)


# In[56]:


# Filter the DataFrame for 'Budget' > X
df_merged = df_merged[df_merged['Budget'] > 47000]
# Check the length of the DataFrame df
df_length = len(df_merged)



# Print the length of the DataFrame
print("Length of DataFrame df:", df_length)
df_merged.to_csv('/Users/danielbrown/Desktop/Golf_Sim_proj_masters.csv', index=False)
df_merged
df = df_merged


# In[57]:


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
#player_counts_df.to_csv('/Users/danielbrown/Desktop/player_counts.csv', index=False)
player_counts_df


# In[58]:


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


# In[ ]:


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


# In[ ]:


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





# In[ ]:




