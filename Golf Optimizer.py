#!/usr/bin/env python
# coding: utf-8

# #Imports necessary libraries
# import pandas as pd
# from pydfs_lineup_optimizer import get_optimizer, Site, Sport, TeamStack, PlayerFilter
# from pydfs_lineup_optimizer import PlayersGroup, PositionsStack, Stack, statistics
# 

# In[ ]:


import pandas as pd
from pydfs_lineup_optimizer import get_optimizer, Site, Sport, TeamStack, PlayerFilter
from pydfs_lineup_optimizer import PlayersGroup, PositionsStack, Stack, statistics


# In[1719]:


#Creates optimizer Object
optimizer = get_optimizer(Site.DRAFTKINGS, Sport.GOLF)
optimizer


# In[1720]:


#optimizer


# In[1721]:


#Loads players into optimizer. This csv must be in the form of csv downloaded from Draftkings. 
optimizer.load_players_from_csv(r"/Users/danielbrown/Desktop/Valspar_Championship_2 - Sheet1.csv")


# In[1722]:


from pydfs_lineup_optimizer import ProgressiveFantasyPointsStrategy
#lineups = optimizer.optimize(n=150, max_exposure=0.60, exposure_strategy=AfterEachExposureStrategy)


# In[1723]:


#optimizer.set_fantasy_points_strategy(ProgressiveFantasyPointsStrategy(0.01))  # Set progressive strategy that increase player points by 1%


# In[1724]:


lineups = optimizer.optimize(n=119)
optimizer.set_fantasy_points_strategy(ProgressiveFantasyPointsStrategy(0.01))  # Set progressive strategy that increase player points by 1%
#optimizer.set_max_repeating_players(3)
#optimizer.set_teams_max_exposures(0.9)


# In[1725]:


#lineups


# In[1726]:


for lineup in lineups:
    print(lineup)


# In[1727]:


optimizer.print_statistic()


# In[ ]:





# In[ ]:





# In[ ]:





# In[1728]:


optimizer.export(r"/Users/danielbrown/Desktop/Draftkings_Golf_DFS/Lineups_Name_IDs/Valspar_Championship_2.csv")


# In[1729]:


df = pd.read_csv(r"/Users/danielbrown/Desktop/Draftkings_Golf_DFS/Lineups_Name_IDs/Valspar_Championship_2.csv")
print(df)   


# In[1730]:


positions = list(df.columns)[:-2]


# In[1731]:


positions


# In[1732]:


fxn = lambda x: x[col].split("(")[1].split(")")[0]


# In[1733]:


for col in positions:
    df[col]=df.apply(fxn,axis=1)


# In[1734]:


df.head(40)


# In[1735]:


df.to_csv(r"/Users/danielbrown/Desktop/Draftkings_Golf_DFS/Lineups_IDs/Valspar_Championship_2.csv")


# In[ ]:





# In[ ]:





# In[ ]:




