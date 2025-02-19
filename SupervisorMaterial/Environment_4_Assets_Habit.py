from gymnasium import Env
from gymnasium.spaces import Discrete, Box
import numpy as np
import random


class Env_4_Assets_Habit(Env):

    # In this environment we have N assets,
    # One of these assets can be the risk-free asset


    def __init__(self,
                 initial_wealth = 1000,
                 initial_habit = 20,
                 lifetime = 200,
                 min_consume_rate = 0.001,
                 max_consume_rate = 0.006,
                 Num_Assets = 4,
                 min_share = np.array([0.1] * 4),
                 max_share = np.array([0.6] * 4),
                 return_mean = np.array([0.05] + [0.15] * (4 - 1)),
                 return_stdev = np.array([0.00] + [0.20] * (4 - 1)),
                 correlations = np.array([[1.0, 0.0, 0.0, 0.0],
                                          [0.0, 1.0, 0.5, 0.3],
                                          [0.0, 0.5, 1.0, 0.3],
                                          [0.0, 0.3, 0.3, 1.0]]),
                 gamma_01 = 3,
                 gamma_02 = 2,
                 habit_growth = 0.05):
        
        # Store the parameters in an instance variable
        self.initial_wealth = initial_wealth
        self.initial_habit = initial_habit
        self.lifetime = lifetime
        self.Num_Assets = Num_Assets
        self.gamma_01 = gamma_01
        self.gamma_02 = gamma_02
        self.habit_growth = habit_growth
        self.return_mean = return_mean
        self.max_share = max_share
        self.min_share = min_share

        # Make covariance matrix from corralations
        covariances = correlations * np.outer(return_stdev, return_stdev)

        # Cholesky does not work if there are riskfree assets. Therefore, the following
        # steps are taken:
        # (1) We have to remove the risk-free assets
        # (2) We calculate the Cholesky matrix
        # (3) We add the risk-free assets back to the covariance matrix at original places
        
        # The columns / rows to be removed
        indices = np.where(np.all(covariances == 0, axis = 1))[0]

        # Remove rows/columns from the array
        arr = np.delete(covariances, indices, axis = 0)
        arr = np.delete(arr, indices, axis = 1)
        
        # Apply Cholesky
        self.cholesky_matrix = np.linalg.cholesky(arr)

        # Add back the rows with zero elements
        Num_Cols = self.cholesky_matrix.shape[1]
        for idx in indices:
            self.cholesky_matrix = np.insert(self.cholesky_matrix, idx, np.zeros(Num_Cols), axis = 0)

        # Add back the columns wtih zero elements
        Num_Rows = self.cholesky_matrix.shape[0]
        for idx in indices:
            self.cholesky_matrix = np.insert(self.cholesky_matrix, idx, np.zeros(Num_Rows), axis = 1)

        print('Correlation')
        print(correlations)
        print('Covariances')
        print(covariances)
        print('Cholesky')
        print(self.cholesky_matrix)

        # The observation space cosnists of the wealth 
        self.observation_space = Box(low=-float(0), high=float('inf'), shape=(1,), dtype=float)
        
        # Action space has two decisions
        # (1) how much to invest in risky asset: between 0 and 1
        # (2) how much wealth to consume: between 0 and 1

        self.action_space = Box(low = np.append(min_share, min_consume_rate), high=np.append(max_share, max_consume_rate), dtype = float)
        
        self.time = None
        
        
    def reset(self, seed = None, options = None):
        
        self.time = 0
        self.wealth_out = self.initial_wealth
        self.habit_out = self.initial_habit
        self.reward = 0

        self.state = np.array([self.initial_wealth])

        # This info is used later
        info = {'Time': self.time,
                'Asset Shares': 'n.a.',
                'Asset Returns': 'n.a.',
                'Rate of Consumption': 'n.a.',
                'Wealth_Out': self.wealth_out,
                'Habit_Out': self.habit_out,
                'Consumption': 0,
                'Utility': 0}
        
        return self.state, info
    
    
    def render(self):
        # Implement visualization if needed
        pass   
    
    
    def step(self, action):
        
        self.share = action[0:self.Num_Assets]
        self.consume_rate = action[-1]

        # Normalize weights
        self.share = np.divide(self.share, np.sum(self.share))

        # Whenever a weight is higher/lower than the allowed weigth, we will immidietly add a penalty
        compare = self.max_share < self.share
        constraint_penalty = compare.astype(int)
        compare = self.min_share > self.share
        constraint_penalty = constraint_penalty + compare.astype(int)
        Penalty = np.sum(constraint_penalty) * self.wealth_out * 0.3      # You may add this as a parameter
        
        # Determination of returns
        standard_normal_vector = np.random.normal(size = self.Num_Assets)
                      
        self.ret = self.cholesky_matrix @ standard_normal_vector 
        
        self.ret = self.ret + self.return_mean

        self.wealth_in = self.wealth_out
        self.wealth_out = self.wealth_in + self.share @ self.ret * self.wealth_in - self.consume_rate * self.wealth_in
        
        self.consumption = self.consume_rate * self.wealth_in

        # Note that consumption is done
        self.habit_in = self.habit_out  
        self.habit_out = self.habit_in * (1 + self.habit_growth)
        
        if self.consumption < self.habit_in:
            self.gamma = self.gamma_01
        else:
            self.gamma = self.gamma_02
        
        # We have to avoid division by zero and division close to Zero
        # Habit is assumed to be positive if externally given
        # Consumption however become Zero => Division by exactly or almost Zero if gamma > 1
        # Divisioon close to Zero will make very large numbers => overflow.
        
        if self.gamma > 1 and self.consumption <= 0.0001 * self.wealth_in:
            self.consumption = 0.000001
            
        self.utility = 1/(1 - self.gamma) * ((self.consumption/self.habit_in)**(1 - self.gamma) - 1) 
        
        self.time = self.time + 1
        
        self.state = np.array([self.wealth_out])
        
        self.reward = self.utility - Penalty
 

        if self.time == self.lifetime:        
            done = True
        else:
            done = False
        
        
        # Information
        info = {'Time': self.time,
                'Asset Shares': self.share,
                'Asset Returns': self.ret,
                'Rate of Consumption': self.consume_rate,
                'Wealth_Out': self.wealth_out,
                'Habit_Out': self.habit_out,
                'Consumption': self.consumption,
                'Utility': self.utility}
      

        
        # Return step information
        return self.state, self.reward, done, False, info           
