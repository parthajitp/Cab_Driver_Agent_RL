# Import routines

import numpy as np
import math
import random


from datetime import datetime
from itertools import product

# Defining hyperparameters
m = 5 # number of cities, ranges from 1 ..... m
t = 24 # number of hours, ranges from 0 .... t-1
d = 7  # number of days, ranges from 0 ... d-1
C = 5 # Per hour fuel and other costs
R = 9 # per hour revenue from a passenger



class CabDriver():

    def __init__(self):
        """initialise your state and define your action space and state space"""
        # All possible combinations of start & drop location plus (0,0) indicating the cab is offline
        self.action_space = [(i,j) for i in range(m) for j in range(m) if i!=j or j == 0] 
        # All possible combinations of location, time and day
        self.state_space = [[loc, time, day] for loc in range(m) for time in range(t) for day in range(d)]
        # A random state from state_space
        self.state_init = random.choice(self.state_space)

        # Start the first round
        self.reset()


     ## Encoding state (or state-action) for NN input

    def state_encod_arch1(self, state):
        """convert the state into a vector so that it can be fed to the NN. This method converts a given state into a vector format. Hint: The vector is of size m + t + d."""
        state_encod = [0 for i in range(m + t +d)]
        # 0 index of state vector is location
        state_encod[int(state[0])] = 1
        # 1 index of state vector is time
        state_encod[m + int(state[1])] = 1
        # 2 index of state vector is day
        state_encod[m + t + int(state[2])] = 1
        return state_encod


    # Use this function if you are using architecture-2 
    def state_encod_arch2(self, state, action):
        """convert the (state-action) into a vector so that it can be fed to the NN. This method converts a given state-action pair into a vector format. Hint: The vector is of size m + t + d + m + m."""
        (x,t,d) , (p,q)
        return state_encod


    ## Getting number of requests

    def requests(self, state):
        """Determining the number of requests basis the location. 
        Use the table specified in the MDP and complete for rest of the locations"""
        location = state[0]
        if location == 0:
            requests = np.random.poisson(2)
        if location == 1:
            requests = np.random.poisson(12)
        if location == 2:
            requests = np.random.poisson(4)
        if location == 3:
            requests = np.random.poisson(7)
        if location == 4:
            requests = np.random.poisson(8)

        if requests >15:
            requests = 15

        # We want to exclude (0, 0) as possible action because it will have to considered by default		
        possible_actions_index = random.sample(range(1, (m-1)*m +1), requests) + [0] 
        actions = [self.action_space[i] for i in possible_actions_index]

        
        actions.append((0,0))

        return possible_actions_index,actions   



    def reward_func(self, state, action, Time_matrix):
        """Takes in state, action and Time-matrix and returns the reward"""
        
        if action == (0, 0):
            rewards = -C
        else:
            time_to_pickup, time_to_dropoff = self.cal_time(state, action, Time_matrix)
            rewards = R * time_to_dropoff - C *(time_to_pickup + time_to_dropoff)
        return rewards


    def cal_time(self, state, action, Time_matrix):
        """Calculates time taken from Cab's current location to pickup location and from pickup location to the dropoff location"""

        cur_loc, cur_time, cur_day = state
        pickup_loc, dropoff_loc = action
        if int(cur_time) == 24:
            cur_time = 0
            if cur_day < 6:
               cur_day += 1
            else:
                cur_day = 0
            
        time_to_pickup = Time_matrix[int(cur_loc), int(pickup_loc), int(cur_time), int(cur_day)]
        time_to_dropoff = Time_matrix[int(pickup_loc), int(dropoff_loc), int(cur_time), int(cur_day)]
        return time_to_pickup, time_to_dropoff

    def next_state_func(self, state, action, Time_matrix):
        """Takes state and action as input and returns next state"""
        cur_loc, cur_time, cur_day = state
        pickup_loc, dropoff_loc = action
        time_to_pickup, time_to_dropoff = self.cal_time(state, action, Time_matrix)
        
        if action != [0, 0]:
            next_loc = dropoff_loc
        else:
            next_loc = cur_loc        
        
        total_time = time_to_pickup + time_to_dropoff
        if total_time == 0 and cur_time < 23:
            next_time = cur_time + 1
            next_day = cur_day
        elif total_time == 0 and cur_time == 23:
            next_time = 0
            if cur_day < 6:
               next_day = cur_day + 1
            else:
                next_day = 0
        elif cur_time + total_time <= 23:
            next_time = cur_time + total_time
            next_day = cur_day
        else:
            next_time = (cur_time + total_time) % 23
            if cur_day < 6:
               next_day = cur_day + 1
            else:
                next_day = 0
        
        next_state = [next_loc, next_time, next_day]
        
        return next_state




    def reset(self):
        return self.action_space, self.state_space, self.state_init
