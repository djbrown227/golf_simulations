#!/usr/bin/env python
# coding: utf-8

# In[126]:


import pandas as pd


# In[127]:


#Import in Probability of Top10 and Missing Cut
df = pd.read_csv(r"/Users/danielbrown/Desktop/DKSalaries-8.csv")
df_2 = pd.read_excel(r"/Users/danielbrown/Desktop/prob_masters.xlsx")
df_2


# In[128]:


df_2 = df_2[['Name','Prob Top 10','Prob Make Cut']]
# Converting percentage columns to decimals
#df_prob['Win%'] = df_prob['Win%'].str.rstrip('%').astype('float') / 100
df_2['Prob Top 10'] = df_2['Prob Top 10'].str.rstrip('%').astype('float') / 100
df_2['Prob Make Cut'] = df_2['Prob Make Cut'].str.rstrip('%').astype('float') / 100
df_2


# In[129]:


# Applying Min-Max normalization
df_2['Prob Top 10'] = (df_2['Prob Top 10'] - df_2['Prob Top 10'].min()) / (df_2['Prob Top 10'].max() - df_2['Prob Top 10'].min())
df_2['Prob Make Cut'] = (df_2['Prob Make Cut'] - df_2['Prob Make Cut'].min()) / (df_2['Prob Make Cut'].max() - df_2['Prob Make Cut'].min())

# Sorting by 'Prob Top 10' in descending order
df_2 = df_2.sort_values('Prob Top 10', ascending=False)

df_2


# In[130]:


df.drop('AvgPointsPerGame', axis=1, inplace=True)
df


# In[131]:


'''

df_2['Points Top 10'] = 60
df_2['Points Make Cut'] = 40
df_2['AvgPointsPerGame'] = round(df_2['Prob Top 10'] * df_2['Points Top 10'] + df_2['Prob Make Cut'] * df_2['Points Make Cut'], 2)
df_2 = df_2[['Name', 'AvgPointsPerGame']]
df_2 = df_2.sort_values(by='AvgPointsPerGame', ascending=False)
df_2.to_csv('/Users/danielbrown/Desktop/Draftkings_Golf_DFS/Prob_T10__MC_Proj/Valspar_Championship_2.csv', index=False)

df_2

'''


# In[132]:


df_joined = pd.merge(df, df_2, on='Name', how='left')
# Dropping any rows with NaN values
df_joined = df_joined.dropna()
df_joined


# In[133]:


df_joined['AvgPointsPerGame'] = (1.2*df_joined['Prob Top 10']+0.8*df_joined['Prob Make Cut'])/df_joined['Salary']*100000
df_joined


# In[134]:


df_joined.to_csv('/Users/danielbrown/Desktop/Player_Projections_Masters_1.csv', index=False)


# In[ ]:





# In[135]:


import pandas as pd
from pydfs_lineup_optimizer import get_optimizer, Site, Sport, TeamStack, PlayerFilter
from pydfs_lineup_optimizer import PlayersGroup, PositionsStack, Stack, statistics


# In[136]:


#Creates optimizer Object
optimizer = get_optimizer(Site.DRAFTKINGS, Sport.GOLF)
optimizer


# In[137]:


#Loads players into optimizer. This csv must be in the form of csv downloaded from Draftkings. 
optimizer.load_players_from_csv(r"/Users/danielbrown/Desktop/Player_Projections_Masters_1.csv")


# In[138]:


from pydfs_lineup_optimizer import ProgressiveFantasyPointsStrategy
#lineups = optimizer.optimize(n=150, max_exposure=0.60, exposure_strategy=AfterEachExposureStrategy)


# In[139]:


lineups = optimizer.optimize(n=200)
#optimizer.set_fantasy_points_strategy(ProgressiveFantasyPointsStrategy(0.01))  # Set progressive strategy that increase player points by 1%
optimizer.set_max_repeating_players(2)
#optimizer.set_teams_max_exposures(0.9)


# In[140]:


for lineup in lineups:
    print(lineup)


# In[141]:


optimizer.print_statistic()


# In[142]:


optimizer.export(r"/Users/danielbrown/Desktop/masters_3.csv")


# In[143]:


df = pd.read_csv(r"/Users/danielbrown/Desktop/masters_3.csv")
print(df) 


# In[144]:


positions = list(df.columns)[:-2]


# In[145]:


positions


# In[146]:


fxn = lambda x: x[col].split("(")[1].split(")")[0]


# In[147]:


for col in positions:
    df[col]=df.apply(fxn,axis=1)


# In[148]:


df.head(40)


# In[149]:


df.to_csv(r"/Users/danielbrown/Desktop/masters_4.csv")


# In[ ]:





# In[ ]:





# In[ ]:




