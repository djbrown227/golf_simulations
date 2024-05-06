### Golf Simulator Overview ###
Below are descriptions of the main two DFS Draftkings Golf Simulators that I use. They are both considered Monte Carlo simulators. 
The first one, Player_Prob_Distributions.py, relies on a custom built hole by hole
simulator, relying on golf hole analysis and clustering, and players individual probability distributions on those holes. We simulate each hole for each player, simulate a
CUT, and end up with a DFS score. We then run the score through a lineup optimizer. We do this 1000's of times and end up with high upside lineups. 

The second one, Golfer_MissMake__Cut_Distributions.py, relies on Top 10, Miss/Make Cut probabilities taken from the PGA website and also player DFS distributions from their
previous tournaments. We now have DFS and probabilties associated with being Top10, Miss/Make Cut. Given this we can simulate tournaments, run those through an optimizer
and find high upside lineups.

**Player_Prob_Distributions.py**
The provided code simulates multiple rounds of a golf tournament for daily fantasy sports (DFS) and implements a cut line based on fantasy points earned by players. Here's a breakdown of what each part of the code does:

1. **Conversion of Points to Strokes and Accumulation**:
   - The `convert_points_to_strokes_and_accumulate` function converts fantasy points earned by players to strokes based on predefined mappings and accumulates strokes for each player across rounds.

2. **Applying Bonuses**:
   - The `apply_bogey_free_round_bonus` function adds bonus points for players who achieve a bogey-free round, enhancing their total points.
   - The `apply_bonus_points` function applies bonus points for players who score three consecutive birdies, enhancing their total points.

3. **Simulation with Cut Line**:
   - The `simulate_multiple_4_rounds_with_cut_line` function simulates multiple rounds of a golf tournament with a cut line after the second round.
   - It iterates through each round, simulating player performance based on probabilities of scoring points on each hole.
   - After the second round, it determines the players to cut based on their cumulative points earned, cutting approximately half of the field.
   - It returns cumulative totals of points earned by players and data for all simulated rounds.

4. **Example Usage**:
   - The code provides an example usage of the `simulate_multiple_4_rounds_with_cut_line` function, where it specifies the number of players to cut and the number of simulations to run.

The provided functions and simulation logic enable the simulation of golf tournaments for DFS, incorporating cut lines based on player performance and fantasy points earned. This allows for realistic simulations of tournaments and the determination of player standings based on fantasy scoring criteria.




**Golfer_MissMake__Cut_Distributions.py**

### Golf Performance Simulation with Player Probabilities ###

The following code snippet performs a simulation of golfer performance based on player statistics and probabilities. Here's an overview of the functionalities:

1. **Merging Dataframes**:
   - The code merges two dataframes, `df_prob` containing probabilities and `combined_stats` containing player statistics, on golfer names to align the probabilities with their stats.

2. **Simulation Function**:
   - The `simulate_golfer_performance` function simulates the performance of a golfer by generating random scores based on their statistics and probabilities.
   - It determines whether the golfer makes the cut based on the probability of making the cut (`Prob Make Cut`).
   - If the golfer makes the cut, it generates a random score based on `Mean_Rd4` and `Std_Rd4`, and calculates the probability of being in the top 10 given that they made the cut.
   - If the golfer does not make the cut, it generates a random score based on `Mean_Rd2` and `Std_Rd2`.

3. **Simulation Across all Golfers**:
   - The `simulate_round` function applies the simulation of golfer performance across all golfers in the dataframe.
   - It utilizes the `simulate_golfer_performance` function to simulate the performance of each golfer.

4. **Simulating the Round**:
   - The code simulates the round by applying the `simulate_round` function to the merged dataframe (`simulation_df`).
   - It sorts the simulated data based on the average points per game (`AvgPointsPerGame`) in descending order.

This functionality enables the simulation of golf tournament rounds based on player statistics and probabilities, providing insights into potential player performances.


### The Rest of the Code ###
  -The other code is will be organized in the future, but it is a mixture of Feature Engineering for DFS, and lineup filtering tools which enable 
  a way of narrowing down through Salary and Diversity metrics the final lineups we want to choose for the competitions.
--- 
