from pymdp.agent import Agent
from pymdp import utils
import numpy as np
from envs.redbluedoors_env.ma_redbluedoors_env import RedBlueDoorEnv
from agents.ql import QLearningAgent
from envs.redbluedoors_env.utils import visualize_trajectories
from envs.redbluedoors_env.ma_redbluedoors_env import RedBlueDoorEnv
from agents.ql import QLearningAgent
from agents.random import RandomAgent  
from agents.aif import ActiveInferenceAgent
from utils.reward_plot import plot_cumulative_avg_rewards





# Configuration
CONFIG_PATH = "envs/redbluedoors_env/configs/config.json"
Q_TABLE_PATH = "q_table.json"
EPISODES = 10  # Number of training episodes
MAX_STEPS = 50  # Maximum steps per episode



# # state (x,y): S_x (column: 0,1,2), S_y (row: 0,1,2), red_door_opened(0,1), blue_door_opened(0,1)

# num_states = [3,3,2,2] #list of dimensionality of each hidden state factor
# num_factors = len(num_states) # number of hidden state factors

# num_controls = [2,2,1,1] # list of dimensionality of each control state factor (how many actions do we have in that state factor)
# num_control_factors = len(num_controls)





# # B : P(S'|S,A)
# B = utils.initialize_empty_B(num_states, num_controls) # [factor][state,state,action]

# for f, ns in enumerate(num_states):
#     B[f] = np.zeros((ns, ns, num_controls[f])) # [state, state, action]
#     if f == 0 or f == 1:
#         # move left/up
#         B[f][0,0:2,0] = 1.0
#         B[f][1,2,0] = 1.
        
#         # move right/down
#         B[f][1,0,1] = 1.0
#         B[f][2,1:,1] = 1.0
    
#     elif f == 2 or f == 3:
#         B[f][1,0,0] = 1.0
#         B[f][1,1,0] = 1.0
        
        
# # utils.plot_likelihood(B[2][:,:,0], "Open")  
# x_pos = [0,1,2]
# y_pos = [0,1,2]
# red_door = [0,1]
# blue_door = [0,1]
# # reward = [0,1] #??????


# # num_obs = [9,2,2] # list of dimensionality of each observation factor
# # A_m_shapes = [ [o_dim] + num_states for o_dim in num_obs]
# # A = utils.obj_array_zeros(A_m_shapes)# 

# # A : P(O|S)
# num_obs = [3,3,2,2] # list of dimensionality of each observation factor
# num_modalities = len(num_obs) # number of observation factors

# A = utils.initialize_empty_A(num_obs, num_states) 
# # print(A.shape)
# A[0] = np.zeros((3, 3, 3, 2, 2))
# A[1] = np.zeros((3, 3, 3, 2, 2))
# A[2] = np.zeros((2, 3, 3, 2, 2))
# A[3] = np.zeros((2, 3, 3, 2, 2))

# # Modality 0: observe agent's x-position (depends on state factor 0)
# for x in range(num_states[0]):
#     A[0][x, x, :, :, :] = 1.0

# # Modality 1: observe agent's y-position (depends on state factor 1)
# for y in range(num_states[1]):
#     A[1][y, :, y, :, :] = 1.0

# # Modality 2: red door (open/closed), depends on state factor 2
# for red in range(num_states[2]):
#     A[2][red, :, :, red, :] = 1.0

# # Modality 3: blue door (open/closed), depends on state factor 3
# for blue in range(num_states[3]):
#     A[3][blue, :, :, :, blue] = 1.0

# for f, no in enumerate(num_obs):
#     if f ==0 or f ==1:
#         A[f] = np.zeros((3, 3, 3, 2, 2))
     
#     elif f == 2 or f == 3:
#         A[f] = np.zeros((2, 3, 3, 2, 2))
        
# utils.plot_likelihood(A[0][:,:,0,0,0], "x-position")
# utils.plot_likelihood(A[1][:,0,:,0,0], "y-position")
# utils.plot_likelihood(A[2][:,:,0,0,0], "red_door")


# D = utils.obj_array(num_factors)
# D[0] = np.array([1., 0., 0.])
# D[1] = np.array([1., 0., 0.])
# D[2] = np.array([1., 0.])
# D[3] = np.array([1., 0.])



# C = utils.obj_array_zeros(num_obs)
# C[0] = np.array([0.5 , 0. , 0.5])
# C[1] = np.array([0.5 , 0. , 0.5])
# C[2] = np.array([ 0., 1.])
# C[3] = np.array([ 0., 1.])



actions = ["up", "down", "left", "right", "open"]

A = ActiveInferenceAgent.generate_matrix_A()
B = ActiveInferenceAgent.generate_matrix_B()
C = ActiveInferenceAgent.generate_matrix_C()
D = ActiveInferenceAgent.generate_matrix_D()



# Initialize environment
env = RedBlueDoorEnv(config_path=CONFIG_PATH)

# Initialize agents
random_agent = RandomAgent(env)
aif_agent = ActiveInferenceAgent(actions, A,B,C,D,1 )

# Track rewards
episode_rewards = []

# Training loop
for episode in range(EPISODES):
    obs, _ = env.reset()
    total_reward = 0
    trajectories = []
    
    for step in range(MAX_STEPS):
        actions = {
            "agent_0": aif_agent.choose_action(obs, "agent_0"),
            # "agent_1": random_agent.choose_action(obs, "agent_1"),
            "agent_1": 4,
            
        }
        print("Obs:\n",obs)
        print("Actions:\n",actions)
        next_obs, rewards, dones, _ = env.step(actions)

        total_reward += sum(rewards.values())
        obs = next_obs

        # Save trajectory for visualization
        trajectories.append(env.render())

        if all(dones.values()):
            break

    # Store total reward per episode
    episode_rewards.append(total_reward)


    if episode % 500 == 0:
        print(
            f"Episode {episode}, Total Reward: {total_reward}"
        )

# Save Q-table
# ql_agent.save_q_table()

# Close environment
env.close()



plot_cumulative_avg_rewards(episode_rewards)