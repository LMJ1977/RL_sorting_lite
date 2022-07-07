### RL Classifier  using a simple Q-learning algorithm in a custom environment
# https://www.gymlibrary.ml/content/environment_creation/
# https://medium.com/@apoddar573/making-your-own-custom-environment-in-gym-c3b65ff8cdaa
# import required libraries
#from curses.panel import bottom_panel
from operator import ne
from cv2 import correctMatches
import numpy as np
import random
import gym
from gym import error, spaces, utils
from gym.utils import seeding
from psutil import boot_time
import time

n_rows = 3
barrier_distance = 3
paddle1_gap = 5
paddle1_to_paddle2_gap = 10
paddle2_to_end_gap = 5
n_cols = barrier_distance + paddle1_gap + paddle1_to_paddle2_gap + paddle2_to_end_gap
element_states = [0, 1, 2, 3]
balls_classified = 0
balls_classified_correct = 0

# available actions are:
#barrier1 = [0,1] # 1: activated, 0: deactivated
barrier2 = [0,1]
#barrier3 = [0,1]
#paddle1_states = [0, 1, 2] # 0: rest, 1: partially up, 2: fully up
#paddle2_states = [0, 1, 2]
paddle1_counter = 0 # for tracking position of paddle1
paddle2_counter = 0 # for tracking position of paddle2
paddle1_actions = [0, 1, 2] # 0: not moving, 1: decreasing, 2: increasing
paddle2_actions = [0, 1, 2]

# define the action array:
        # action_array[0] = barrier1 action
        # action_array[1] = barrier2 action
        # action_array[2] = barrier3 action
        # action_array[3] = paddle1 action
        # action_array[4] = paddle2 action
#initialize the action array as a 5 element list
action_array = [0, 0, 0]

# define the action space
#action_space = spaces.MultiDiscrete([[0,1], [0,1], [0,1], [0,2], [0,2]])
action_space = spaces.Discrete(2 * 3**2)
# create a list wit column number with barrier elements
barrier_cols = [barrier_distance]
# create a list with column number with paddle1  partially deployed elements
paddle1_partially_deployed_cols = [barrier_distance+paddle1_gap]
# create a list with column number with paddle1 fully deployed elements
paddle1_fully_deployed_cols = [barrier_distance+paddle1_gap+1]
# create a list with column number with paddle2 partially deployed elements
paddle2_partially_deployed_cols = [barrier_distance+paddle1_gap+paddle1_to_paddle2_gap]
# create a list with column number with paddle2 fully deployed elements
paddle2_fully_deployed_cols = [barrier_distance+paddle1_gap+paddle1_to_paddle2_gap+1]


# define the environment
class BallSortingEnv_lite(gym.Env):
    """
    Ball sorting environment
    by L.Monzon , ITAINNOVA, 2022

    ### Description
    System is a 3 channel conveyor belt with balls.
    The balls are sorted in 3 channels, red, green and blue.
    The system has a fixed size grid trying to move 1 position to the right each time step.
    For each position in the grid, being able to move to the rigth is only possible if there is a void position to the right.
    In order to move, system moves to the right starting from the rightmost position.
    Classifier tries to sort balls, red in the first row, green in the second row, blue in the third row.
    System gets reward of +10 for each correct ball placement, -5 for each incorrect ball placement, -1 for paddle action -1 if paddle state is 0  and 0 for each void
    enviroment has the following 

    ### Actions:
    -   activating a stop barrier at step barriers_distance, where barriers_distance is a user defined number. Barrier works independently for each row
            when a barrier is activated, the movement of the balls is stopped for that row before the barrier but not after the barrier
    -   changing paddle1 state
            Paddle 1 can be used to move the balls.
            Paddle 1 state can be 0: rest, 1 and 2.
            Paddle 1 state 0 does not move the balls.
            Paddle 1 state 1 moves the ball from the last row 1 position up for the next step
            Paddle 1 state 2 moves the ball from the last row 1 position up and the ball in the center row 1 position up for the next step
            Paddle 1 action can be 0: keep state, +1, increment state, -1, decrement state.
    -   changing paddle2 stat
            Paddle 2 can be used to move the balls.
            Paddle 2 state can be 0: rest, 1 and 2.
            Paddle 2 state 0 does not move the balls.
            Paddle 2 state 1 moves the ball from the first row 1 position down for the next step
            Paddle 2 state 2 moves the ball from the first row 1 position down and the ball in the center row 1 position down for the next step
            Paddle 2 action can be 0: keep state, +1, increment state, -1, decrement state.

    ### Observations:
    -  stop barriers are placed at step barriers_distance, where barriers_distance is a user defined number. Barrier works independently for each row
    -  balls can only move to the right if the target position is empty, otherwise they are stopped
    -  balls can only move up and down if the target position is empty, otherwise they are stoppedº

    ### Reward:
    - reward is evaluated at last column of the grid each time step
    - reward is +10 for each correct ball placement, -5 for each incorrect ball placement, -100 for paddle1 or 2 action -1 if paddle1 or 2 state is 0  and 0 for each void
    - reward is maximized at the end of the episode
    - reward is 0 at the beginning of the episode
    
    ### Episode steps:
    - each time_step last column is evaluated and discarded, all other columns moves to the right following instructions in the next time_step
    - empty positions in the first column are filled with a random number between 0 and 3

    ### Episode Termination:
    -   episode ends after 200 time steps

    ### Version  History
    -   v0.0.1: initial version
https://stackoverflow.com/questions/71978756/keras-symbolic-inputs-outputs-do-not-implement-len-error
"""
    metadata = {'render.modes': ['ansi']}
    
    def __init__(self):
        # initialize enviroment as str array
            
        self.state = np.zeros((n_rows, n_cols), dtype=np.uint8)
        for i in range(n_rows):
                for j in range(n_cols):
                        # random initialization
                        #self.state[i][j] = np.random.choice(element_states,1)[0]               
                        # initialize with 0
                        self.state[i][j] = 0
        self.ministate = self.state[1,0:3]                
        self.steps_remaining = 2000
        self.reward = 0
        self.done = False
        self.info = None
        self.action_space = gym.spaces.Discrete(2*3*3)
        self.observation_space = gym.spaces.Box(0,3,[1,3], dtype=np.uint8)#[ n_rows, n_cols], dtype=np.uint8)
        self.paddle1_counter = paddle1_counter 
        self.paddle2_counter = paddle2_counter
        self.n_rows = n_rows
        self.n_cols = n_cols
        self.barrier_distance = barrier_distance
        self.barrier_cols = barrier_cols
        self.paddle1_gap = paddle1_gap
        self.paddle1_to_paddle2_gap = paddle1_to_paddle2_gap
        self.paddle1_partially_deployed_cols = paddle1_partially_deployed_cols
        self.paddle1_fully_deployed_cols = paddle1_fully_deployed_cols
        self.paddle2_partially_deployed_cols = paddle2_partially_deployed_cols
        self.paddle2_fully_deployed_cols = paddle2_fully_deployed_cols
        self.action_array = action_array
        self.balls_classified = balls_classified
        self.balls_classified_correct = balls_classified_correct
        self.tasa_aciertos = 0
        self.reset()
    
#     def check_action(self, action_array):
#             # checking if paddle actions are valid
#             # if not, correct and give penalty
#                 if action_array[3] == 1:
#                         if self.paddle1_counter == 0:
#                                 action_array[3] = 0
#                                 self.reward -= 10
#                 elif action_array[3] == 2:
#                         if self.paddle1_counter == 2:
#                                 action_array[3] = 0
#                                 self.reward -= 10
#                 #if ok, update paddle1 counter
#                 else: self.paddle1_counter += action_array[3]
#                 #checking if paddle2 action is valid
#                 if action_array[4] == 1:
#                         if self.paddle2_counter == 0:
#                                 action_array[4] = 0
#                                 self.reward -= 10
#                 elif action_array[4] == 2:
#                         if self.paddle2_counter == 2:
#                                 action_array[4] = 0
#                                 self.reward -= 10
#                 #if ok, update paddle2 counter
#                 else: self.paddle2_counter += action_array[4]
#                 return action_array


    def check_action(self):
        # checking if paddle actions are valid
        # if not, correct and give penalty
        if self.action_array[-2] == 1:
                if self.paddle1_counter == 0:
                        #action_array[3] = 0
                        #pass
                        self.reward -= 10.0
                else: 
                        if self.action_array[-2] == 1:
                                self.paddle1_counter -= 1
                        elif self.action_array[-2] == 2:
                                self.paddle1_counter +=1
        elif self.action_array[-2] == 2:
                if self.paddle1_counter == 2:
                        #action_array[3] = 0
                        #pass
                        self.reward -= 10.0
        #if ok, update paddle1 counter
                else: 
                        if self.action_array[-2] == 1:
                                self.paddle1_counter -= 1
                        elif self.action_array[-2] == 2:
                                self.paddle1_counter +=1

        #checking if paddle2 action is valid
        if self.action_array[-1] == 1:
                if self.paddle2_counter == 0:
                        #action_array[4] = 0
                        self.reward -= 1.0
                else: 
                        if self.action_array[-1] == 1:
                                self.paddle2_counter -=1
                        elif self.action_array[-1] == 2:
                                self.paddle2_counter +=1
        elif self.action_array[-1] == 2:
                if self.paddle2_counter == 2:
                        #action_array[4] = 0
                        self.reward -= 1.0
        #if ok, update paddle2 counter
                else: 
                        if self.action_array[-1] == 1:
                                self.paddle2_counter -=1
                        elif self.action_array[-1] == 2:
                                self.paddle2_counter +=1
                
    def step(self, action):
        self.action_array = self.decode_action(action)
        # print(self.action_array)
        # check if the action is valid
        self.check_action()      
        self.update_state()
        self.steps_remaining -= 1
        self.evaluate_reward()
        self.ministate = self.state[1,0:3]
        #self.done = self.done
        #self.info = {"tasa_aciertos": self.balls_classified_correct/self.balls_classified}
        #self.render()
        #return  self.encode_state(), self.reward, self.end_episode(), {"tasa_aciertos": self.tasa_aciertos}
        return  self.ministate, self.reward, self.end_episode(), {"tasa_aciertos": self.tasa_aciertos}

    def evaluate_reward(self):
        # check last column for correct ball placement
        # if correct, reward is +40 for each correct ball placement
        # if void, reward is 0
        # if incorrect, reward is -5 for each incorrect ball placement
        for i in range(n_rows):
                if self.state[i][n_cols-1] != 0:
                        self.balls_classified += 1
                        #self.reward += 1
                else:
                        #self.state[i][n_cols-1] == 0:
                        self.reward -= 1.0
                if self.state[i][n_cols-1] == i+1:
                        self.reward += 40.0
                        self.balls_classified_correct += 1

                else:
                        self.reward -= 500.0
                        #pass
        try:                 
                self.tasa_aciertos = self.balls_classified_correct/self.balls_classified
        except ZeroDivisionError:
                self.tasa_aciertos = 0
        #self.reward += (self.tasa_aciertos-0.3333)*10 
    def update_state(self):
        def move_1(self,arr, dir):
                #create a new array like arr
                new_arr = [0,0,0]
                new_curr_arr = [0,0,0]   
                #if dir is 1, reverse  array
                if dir == 1:
                        arr = arr[::-1]      
                if arr[0] == 0:
                        new_arr = arr
                        new_curr_arr = [0,0,0]
                else:
                        if arr[1] == 0:
                                if arr[0] != 0:
                                        new_arr = [0,arr[0],arr[2]]
                                        new_curr_arr = [0,0,0]
                                else:
                                        new_arr = [0,arr[1],arr[2]]
                                        new_curr_arr = [arr[0],0,0]
                        else: 
                                if arr[2] == 0:
                                        if arr[0] != 0:
                                                new_arr = [0,arr[0],arr[1]]
                                                new_curr_arr = [0,0,0]
                                        else:
                                                new_arr = [0,arr[1],arr[2]]
                                                new_curr_arr = [arr[0],0,0]
                                else:
                                        new_arr =  [0,arr[1],arr[2]]
                                        new_curr_arr = [arr[0],0,0]
                #if dir is 1, reverse  array
                if dir == 1:
                        new_arr = new_arr[::-1]
                        new_curr_arr = new_curr_arr[::-1]
                return new_curr_arr, new_arr

        def move_2(self, arr, dir):
                #create a new array like arr
                new_arr = [0,0,0]
                new_curr_arr = [0,0,0]   
                #if dir is 0, return arr
                if dir == 1:
                        arr = arr[::-1]
                if arr[1] == 0:
                        new_arr = arr
                        new_curr_arr = [0,0,0]
                elif arr[2] == 0:
                        new_arr = [0,0,arr[1]]
                        new_curr_arr = [0,0,0]
                else:
                        new_arr = [0,0,arr[2]]
                        new_curr_arr = [arr[0],arr[1],0]
                #if dir is 1, reverse  array
                if dir == 1:
                        new_arr = new_arr[::-1]
                        new_curr_arr = new_curr_arr[::-1]
                return new_curr_arr,new_arr  
                
        state = self.state.copy()
        #change last column values to voids
        for i in range(n_rows):
                state[i][n_cols-1] = 0
        #move balls to the right if possible, col by col from right to left
        for j in range(n_cols-2, -1, -1):
                # check if there are barrier or paddles in the current column
                if j in barrier_cols:
                        #loop rows
                        for row in range(n_rows):
                        # check if the barrier action is 1, if so, stop the balls in the current column
                                #if self.action_array[row] != 1:
                                if row == 1 and self.action_array[0] != 1:
                                        # move current ball
                                        state[row][j+1] = state[row][j]
                                        state[row][j] = 0
                ### PADDLE ACTIONS                        
                # chek if j in paddle1_partially_deployed_cols
                elif j in self.paddle1_partially_deployed_cols and self.paddle1_counter >=1:
                        state[:,j], state[:,j+1] = move_1(self, state[:,j], 1) # 1 for reverse
                # check if j in paddle2_partially_deployed_cols
                elif j in self.paddle2_partially_deployed_cols and self.paddle2_counter >=1:
                        state[:,j], state[:,j+1] = move_1(self, state[:,j], 0)
                # check if j in paddle1_fully_deployed_cols
                elif j in self.paddle1_fully_deployed_cols and self.paddle1_counter > 1:
                        state[:,j], state[:,j+1] = move_2(self, state[:,j], 1)
                # check if j in paddle2_fully_deployed_cols
                elif j in self.paddle2_fully_deployed_cols and self.paddle2_counter > 1:
                        state[:,j], state[:,j+1] = move_2(self, state[:,j], 0)
                else: # move ball normally, checking if there is void in next column
                        for i in range(n_rows):
                                if state[i][j+1] == 0:
                                        state[i][j+1] = state[i][j]
                                        state[i][j] = 0
        # add new balls to the first column if necessary
        # check for voids in the first column
        # if there is void, add a ball using random choice to the first column
        # if there is no void, do not add a ball
        #loop rows
        for i in range(n_rows):
                if state[i][0] == 0:
                        if i == 1:
                                state[i][0] = np.random.choice(element_states,1)[0]
                        else:
                                state[i][0] = 0
        self.state = state


    def encode_state(self):
        # make a copy of the state and flatten it
        state = self.state.copy()
        state = state.flatten()
        encoder = 0
        encoder += state[0]
        for state in state[1:]:
                encoder *= 4
                encoder += state
         
        return encoder
 
    def decode_state(self, encoder):
        # make a copy of the encoder
        enc = encoder.copy()
        out = []
        for _ in range(len(enc[:-1])):
                out.append(enc % 4)
                enc //= 4
        out.append(enc)
        out[::-1]
        return out.reshape(n_rows, n_cols)
        


    def encode_action(self, action_array):
        # (2), 3, 3
        i = 0
        i += action_array[0]
        i *= 3
        i += action_array[1]
        i *= 3
        i += action_array[2]

        return i

    def decode_action(self, i):
        out = []
        out.append(i % 3)
        i = i // 3
        out.append(i % 3)
        i = i // 3
        out.append(i)
        #assert 0 <= i < 5
        return out[::-1]

    def reset(self):
        # initialize grid with random numbers
        self.state = np.zeros((n_rows, n_cols))
        for i in range(n_rows):
                for j in range(n_cols):
                        self.state[i][j] = random.randint(0, len(element_states)-1)
        
        # force (overwrite) first col 0 and 2 positions to be 
        self.state[0][0] = 0
        self.state[2][0] = 0
        self.steps_remaining = 2000
        self.reward = 0
        self.done = False
        self.info = None
        self.paddle1_counter = 0 # for tracking position of paddle1
        self.paddle2_counter = 0 # for tracking position of paddle2
        self.balls_classified = 0
        self.balls_classified_correct = 0
        self.ministate = self.state[1,0:3]
        #print("Environment reset")
        #self.state = self.state.ravel()
        return self.ministate#self.ministate



    def render(self, mode='ansi', close=False):
        def replace_char(str_in,position,new_char):
            return str_in[:position] + new_char + str_in[position+1:]
        # initialize row_list
        row_list = []
        # loop over rows and cols

        for i in range(n_rows):
        # make a string with the row values
                row_str = ''
                for j in range(n_cols):
                        row_str += str(self.state[i][j])
                # add spacers between elements, ':' if not blocked, '|' if blocked. 
        # blocked elements are in barrier_cols if row barrier col value is 1
        # same for paddles 
        # identify block
                        if j in self.barrier_cols and self.action_array[i] == 1:
                                separator_char =  '|'
                        elif j in self.paddle1_partially_deployed_cols and self.paddle1_counter >=1 and i == self.n_rows - 1:
                                separator_char =  '|'
                        elif j in self.paddle2_partially_deployed_cols and self.paddle2_counter >=1 and i == 0:
                                separator_char =  '|'
                        elif j in self.paddle1_fully_deployed_cols and self.paddle1_counter > 1 and i == self.n_rows - 2:
                                separator_char =  '|'
                        elif j in self.paddle2_fully_deployed_cols and self.paddle2_counter > 1 and i == 1:
                                separator_char =  '|'
                        else:
                                separator_char =  ':'
                        row_str += separator_char
        # for each not zero element, replace with '*' and color it,1 red, 2 blue, 3 yellow
                row_str = row_str.replace('0.', ' ')
                row_str = row_str.replace('1.', 'R')
                row_str = row_str.replace('2.', 'B')
                row_str = row_str.replace('3.', 'Y')
                row_str = row_str.replace('0', '')
                row_list.append(row_str)
        # print a line at the top at barrier_cols including separators
        row_len = len(row_list[0])
        top_panel = '-' * row_len
        target_pos = self.barrier_distance*2

        top_panel=replace_char(top_panel,target_pos, 'B')
        # print paddle1
        target_pos = (self.barrier_distance + self.paddle1_gap + self.paddle1_to_paddle2_gap)*2
        top_panel=replace_char(top_panel,target_pos, 'P')
        # carriage return at the beginning of the line
        top_panel = '\r' + top_panel

        #print top_panel
        print(top_panel)
        # print upper border

        #print('-' * row_len) 
        # print environment
        for row in row_list:
                print(row)
        # print lower border
        #print('-' * row_len)
        # print a line at the bottom at barrier_cols including separators
        bottom_panel = '-' * row_len
        target_pos = self.barrier_distance*2
        bottom_panel=replace_char(bottom_panel,target_pos, 'B')
        # print paddle1
        target_pos = (self.barrier_distance + self.paddle1_gap)*2
        bottom_panel=replace_char(bottom_panel,target_pos, 'P')
        print(bottom_panel)
        # pause for a bit for readable output
        time.sleep(0.2)
        

        
    def end_episode(self):
        """
        Returns True if the episode is over.
        """
        if self.steps_remaining <= 0:
                self.render()
                print('{} Bolas clasificadas corretamente'.format(self.balls_classified_correct))
                print('{} Bolas clasificadas'.format(self.balls_classified))
                print('Tasa de acierto: {}%'.format(self.balls_classified_correct/self.balls_classified*100))
                return True
        else:
                return False
        #return self.steps_remaining <= 0

    def close(self):
        if self.window is not None:
                import pygame
                pygame.display.quit()
                pygame.quit()



