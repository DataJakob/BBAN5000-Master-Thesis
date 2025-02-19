from gymnasium import Env
from gymnasium.spaces import Discrete, Box
import numpy as np
import random

class Environment_Ext_Habit(Env):


    def __init__(self,
                 initial_wealth = 1000,
                 initial_habit = 20,
                 lifetime = 200,
                 min_consume_rate = 0.001,
                 max_consume_rate = 0.006,
                 min_risky_share = 0.1,
                 max_risky_share = 0.6,
                 risk_free_rate = 0.02,
                 risky_return_mean = 0.07,
                 risky_return_stdev = 0.20,
                 gamma_01 = 3,
                 gamma_02 = 2,
                 habit_growth = 0.05):
        
        # Store the parameters in an instance variable
        self.initial_wealth = initial_wealth
        self.initial_habit = initial_habit
        self.lifetime = lifetime
        self.risk_free_rate = risk_free_rate       
        self.risky_return_mean = risky_return_mean
        self.risky_return_stdev = risky_return_stdev
        self.gamma_01 = gamma_01
        self.gamma_02 = gamma_02
        self.habit_growth = habit_growth        
        
        # The observation space cosnists of the wealth 
        self.observation_space = Box(low=-float(0), high=float('inf'), shape=(1,), dtype=float)
        
        # Action space has two decisions
        # (1) how much to invest in risky asset: between 0 and 1
        # (2) how much wealth to consume: between 0 and 1

        self.action_space = Box(low = np.array([min_risky_share, min_consume_rate]), high=np.array([max_risky_share, max_consume_rate]), dtype = float)
        
        self.time = None
        
        
    def reset(self, seed = None, options = None):
        
        self.time = 0
        self.wealth_out = self.initial_wealth
        self.habit_out = self.initial_habit
        self.reward = 0

        self.state = np.array([self.initial_wealth])

        # This info is used later
        info = {'Time': self.time,
                'Risky Asset Share': 'n.a.',
                'Risky Asset Return': 'n.a.',
                'Risk-free rate': 'n.a.',
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
        
        self.risky_share = action[0]
        self.consume_rate = action[1]
        
        self.risky_return = self.risky_return_mean + self.risky_return_stdev * random.gauss(0,1)

        
        self.wealth_in = self.wealth_out
        self.wealth_out = self.wealth_in \
                        + (1 - self.risky_share) * self.wealth_in * self.risk_free_rate \
                        + self.risky_share * self.wealth_in * self.risky_return \
                        - self.consume_rate * self.wealth_in
        
        self.consumption = self.consume_rate * self.wealth_in

        # Note that consumption is don
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
        
        self.reward = self.utility
 

        if self.time == self.lifetime:        
            done = True
        else:
            done = False
        
        
        # Information
        info = {'Time': self.time,
                'Risky Asset Share': self.risky_share,
                'Risky Asset Return': self.risky_return,
                'Risk-free rate': self.risk_free_rate,
                'Rate of Consumption': self.consume_rate,
                'Wealth_Out': self.wealth_out,
                'Habit_Out': self.habit_out,
                'Consumption': self.consumption,
                'Utility': self.utility}
        
        # Return step information
        return self.state, self.reward, done, False, info           
