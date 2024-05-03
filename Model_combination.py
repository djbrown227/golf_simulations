#!/usr/bin/env python
# coding: utf-8

# In[28]:


import pandas as pd


# In[29]:


import pandas as pd

# Loading the dataframes
df = pd.read_csv(r"/Users/danielbrown/Desktop/Golf_Sim_proj_masters.csv")
df_1 = pd.read_csv(r"/Users/danielbrown/Desktop/Golf_Sim_masters_2.csv")
df_2 = pd.read_csv(r"/Users/danielbrown/Desktop/masters_3.csv")

# Concatenating the dataframes
combined_df = pd.concat([df, df_1, df_2], ignore_index=True)
df = combined_df


# In[ ]:


df


# In[ ]:


df.to_csv('/Users/danielbrown/Desktop/full_master.csv', index=False)


# In[ ]:


import pandas as pd

# Assuming df is your DataFrame
# Let's say you want to randomly select 5 rows
random_rows = df.sample(n=234)

# Display the randomly selected rows
print(random_rows)

df = random_rows
df.to_csv('/Users/danielbrown/Desktop/full_master_selectedrows.csv', index=False)


# In[ ]:


# Initialize an empty dictionary to count player appearances
player_counts = {}

# Iterate over each column and count occurrences
for col in df.columns:
    if col.startswith('G'):  # Check if the column is one of the player columns
        for player in df[col]:
            if player in player_counts:
                player_counts[player] += 1
            else:
                player_counts[player] = 1

# Convert the player_counts dictionary to a dataframe
player_counts_df = pd.DataFrame(player_counts.items(), columns=['Player', 'Count'])

# Sort the dataframe by 'Count' in descending order
player_counts_df_sorted = player_counts_df.sort_values(by='Count', ascending=False)

# Display the sorted dataframe
print(player_counts_df_sorted)


# In[ ]:


import pandas as pd
import re
from itertools import combinations
from collections import Counter

# Load the CSV file
#df = pd.read_csv('/Users/danielbrown/Desktop/Golf_Sim_output_1.csv')

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


#random sample 234 from the above


# In[ ]:


import pandas as pd

# Assuming df is your DataFrame
# Let's say you want to randomly select 5 rows
random_rows = df.sample(n=234)

# Display the randomly selected rows
print(random_rows)

df = random_rows
df.to_csv('/Users/danielbrown/Desktop/full_master_selectedrows.csv', index=False)


# In[ ]:


# Initialize an empty dictionary to count player appearances
player_counts = {}

# Iterate over each column and count occurrences
for col in df.columns:
    if col.startswith('G'):  # Check if the column is one of the player columns
        for player in df[col]:
            if player in player_counts:
                player_counts[player] += 1
            else:
                player_counts[player] = 1

# Convert the player_counts dictionary to a dataframe
player_counts_df = pd.DataFrame(player_counts.items(), columns=['Player', 'Count'])

# Sort the dataframe by 'Count' in descending order
player_counts_df_sorted = player_counts_df.sort_values(by='Count', ascending=False)

# Display the sorted dataframe
print(player_counts_df_sorted)


# In[ ]:


import pandas as pd
import re
from itertools import combinations
from collections import Counter

# Load the CSV file
#df = pd.read_csv('/Users/danielbrown/Desktop/Golf_Sim_output_1.csv')

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




