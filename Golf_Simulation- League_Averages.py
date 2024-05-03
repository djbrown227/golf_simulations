#!/usr/bin/env python
# coding: utf-8

# In[2]:


#This is a simple model that simulates round scores based on PGA model probability and PGA tour league wide
#average fantasy points.
#Simulates a golfer performance based on average points when making top 10, 
#making cut but not top 10, and missing cut. These are all based on tour averages

#After simulating a tournament, the optimal lineup is found using an lineup optimizer.
#The simulaiton is then run again


# In[1]:


import pandas as pd
import numpy as np
from pydfs_lineup_optimizer import get_optimizer, Site, Sport, TeamStack, PlayerFilter
from pydfs_lineup_optimizer import PlayersGroup, PositionsStack, Stack, statistics


# In[2]:


df = pd.read_csv(r'/Users/danielbrown/Desktop/Draftkings_Golf_DFS/Prob_T10_MC_Sal/Valspar_Championship_2.csv')
df = df[['Name','Prob Top 10', 'Prob Make Cut']]
df


# In[3]:


def simulate_golfer_performance(golfer):
    # Using a copy to avoid SettingWithCopyWarning
    golfer = golfer.copy()
    # Determine if the golfer makes the cut
    make_cut = np.random.rand() < golfer['Prob Make Cut']
    golfer['Made Cut'] = 1 if make_cut else 0
    
    if make_cut:
        # Calculate the conditional probability of being in the top 10, given the golfer makes the cut
        prob_top_10_given_cut = golfer['Prob Top 10'] / golfer['Prob Make Cut']
        #print(prob_top_10_given_cut)
        in_top_10 = np.random.rand() < prob_top_10_given_cut
        golfer['Top 10'] = 1 if in_top_10 else 0
        points = np.random.normal(112.4, 15.22) if in_top_10 else np.random.normal(74.5, 10.67)
    else:
        golfer['Top 10'] = 0
        points = np.random.normal(26.48, 5.68)

    golfer['AvgPointsPerGame'] = max(points, 0)  # Ensure non-negative points
    return golfer

def simulate_round(df):
    # Apply the simulation to each golfer and update the dataframe
    simulated_data = df.apply(simulate_golfer_performance, axis=1)
    return simulated_data

# Simulate a round and display the results
simulated_round = simulate_round(df)
print(simulated_round)


# In[ ]:





# In[4]:


# Exporting the DataFrame to a CSV file
#simulated_round.to_csv('/Users/danielbrown/Desktop/df_simulated_round_summary.csv', index=False)


# In[5]:


df_2 = pd.read_csv(r'/Users/danielbrown/Desktop/Draftkings_Golf_DFS/Player_Projections_Corr/Valspar_Championship_2.csv')
df_2 = df_2.drop('AvgPointsPerGame', axis=1)
df_2


# In[6]:


#simulated_round = simulated_round[['Name','AvgPointsPerGame']]


# In[7]:


# Left join operation
#merged_df = pd.merge(df_2, simulated_round, on='Name', how='left')
#merged_df.to_csv('/Users/danielbrown/Desktop/merged_df_simulation.csv', index=False)


# In[8]:


all_lineups = []
num_iterations = int(input("Enter the number of times to repeat the block: "))


# In[9]:


for _ in range(num_iterations):

    simulated_round = simulate_round(df)
    simulated_round = simulated_round[['Name','AvgPointsPerGame']]

    merged_df = pd.merge(df_2, simulated_round, on='Name', how='left')
    merged_df.to_csv('/Users/danielbrown/Desktop/merged_df_simulation.csv', index=False)

    #Creates optimizer Object
    optimizer = get_optimizer(Site.DRAFTKINGS, Sport.GOLF)
    optimizer
    optimizer.load_players_from_csv(r"/Users/danielbrown/Desktop/merged_df_simulation.csv")
    lineups = optimizer.optimize(n=1)

    for lineup in lineups:
        all_lineups.append(lineup)
    


# In[ ]:





# In[ ]:





# In[10]:


#Creates optimizer Object
#optimizer = get_optimizer(Site.DRAFTKINGS, Sport.GOLF)
#optimizer
#optimizer.load_players_from_csv(r"/Users/danielbrown/Desktop/merged_df_simulation.csv")
#lineups = optimizer.optimize(n=1)


# In[11]:


#all_lineups = []  # Initialize an empty list to store all lineups

#for lineup in lineups:
#    all_lineups.append(lineup)


# In[12]:


# Printing the all_lineups list to verify the appended data
for lineup in all_lineups:
    print(lineup)


# In[13]:


#optimizer.print_statistic()


# In[14]:


#optimizer.export(r"/Users/danielbrown/Desktop/Golf_Sim.csv")


# In[15]:


#df = pd.read_csv(r"/Users/danielbrown/Desktop/Golf_Sim.csv")
#print(df) 
lineup


# In[16]:


lineup_data = []

# Iterate over all_lineups and extract lineup information
for lineup in all_lineups:
    lineup_info = [player.full_name for player in lineup.players]
    lineup_info.extend([lineup.salary_costs, lineup.fantasy_points_projection])
    lineup_data.append(lineup_info)

# Create a DataFrame from the lineup data
df = pd.DataFrame(lineup_data, columns=['G','G','G','G','G','G','Budget','FPPG'])

# Export the DataFrame to a CSV file
df.to_csv('/Users/danielbrown/Desktop/Golf_Sim_output.csv', index=False)


# In[17]:


# Filter the DataFrame for 'Budget' > 45000
df = df[df['Budget'] > 49200]
# Check the length of the DataFrame df
df_length = len(df)



# Print the length of the DataFrame
print("Length of DataFrame df:", df_length)
df.to_csv('/Users/danielbrown/Desktop/Golf_Sim_output_1.csv', index=False)
df


# In[18]:


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
#player_counts_df


# In[19]:


# Create a dictionary mapping player names to names + IDs
name_id_dict = df_2.set_index('Name')['Name + ID'].to_dict()

# Merge the dataframes on the names using the dictionary
df_merged = df.replace(name_id_dict)

# Reset the index of df_merged
df_merged.reset_index(drop=True, inplace=True)

print(df_merged)


# In[20]:


df_merged.to_csv('/Users/danielbrown/Desktop/golf_sim.csv', index=False)


# In[ ]:




