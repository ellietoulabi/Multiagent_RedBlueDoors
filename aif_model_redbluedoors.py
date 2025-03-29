import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pymdp
from pymdp import utils
from pymdp.maths import softmax
from pymdp.agent import Agent
import copy, math, random

np.random.seed(0)
random.seed(0)


# State space
self_pos_state_factor = [
    "pos_0",  # (1,1)
    "pos_1",  # (2,1)
    "pos_2",  # (3,1)
    "pos_3",  # (1,2)
    "pos_4",  # (2,2)
    "pos_5",  # (3,2)
    "pos_6",  # (1,3)
    "pos_7",  # (2,3)
    "pos_8",  # (3,3)
]

doors_state_factor = [
    "both_closed",
    "blue_open_red_closed",
    "red_open_blue_closed",
    "both_open",
]

num_states = [len(self_pos_state_factor), len(doors_state_factor)]
num_factors = len(num_states)


# Observation space
self_pos_observation_modality = [
    "pos_0",  # (1,1)
    "pos_1",  # (2,1)
    "pos_2",  # (3,1)
    "pos_3",  # (1,2)
    "pos_4",  # (2,2)
    "pos_5",  # (3,2)
    "pos_6",  # (1,3)
    "pos_7",  # (2,3)
    "pos_8",  # (3,3)
]
outcome_obervation_modality = [
    "no_progress",      # both doors closed
    "task_failed",      # blue opened before red
    "progress_started", # red opened, blue not
    "task_completed",   # both open
]

num_obs = [len(self_pos_observation_modality), len(outcome_obervation_modality)]
num_modalities = len(num_obs)

door_action_names = ['open']
self_pos_action_names = ['up', 'down', 'left', 'right']

num_controls = [len(self_pos_action_names),len(door_action_names)] # 4 actions (up, down, left, right) and 1 action (open door)
num_control_factors =  len(num_controls)


# A matrix      [for each obs_modality m: A[m][obs_idx, state_idx]= P(obs | state)]
A = utils.initialize_empty_A(num_obs, num_states)

# A[0] for self_pos_observation_modality        self_pos_state → observed position (identity) 
A[0] = np.zeros((num_obs[0], num_states[0], num_states[1]))  # shape: (9, 9, 4)

for i in range(num_states[0]):  # self position
    for j in range(num_states[1]):  # door state (unused here)
        A[0][i, i, j] = 1.0  # identity: P(obs_pos = i | pos_state = i) = 1

# A[1] for outcome_obervation_modality          doors_state → outcome signal
A[1] = np.zeros((num_obs[1], num_states[0], num_states[1]))  # shape: (4, 9, 4)
door_to_outcome = {
    0: 0,  # both_closed → no_progress
    1: 1,  # blue_open_red_closed → task_failed
    2: 2,  # red_open_blue_closed → progress_started
    3: 3   # both_open → task_completed
}

for pos in range(num_states[0]):
    for door_state, outcome in door_to_outcome.items():
        A[1][outcome, pos, door_state] = 1.0


# utils.plot_likelihood(A[1][:,2,:], title = "state mod; both closed; seeing pos if in pos " )
for m in range(num_modalities):
    print(f"A[{m}] normalized: {utils.is_normalized(A[m])}")
    # print(A[0])



# B matrix      [for each state factor f: B[f][state_idx, state_idx, action]= P(state' | state, action)]
B = utils.initialize_empty_B(num_states, num_controls) # [factor][state,state,action]

for f, ns in enumerate(num_states):
    B[f] = np.zeros((ns, ns, num_controls[f])) # [state, state, action]


# B[0] for self_pos_state_factor  

pos_idx_to_xy = {
    0: (0,0), 1: (1,0), 2: (2,0),
    3: (0,1), 4: (1,1), 5: (2,1),
    6: (0,2), 7: (1,2), 8: (2,2)
}
xy_to_pos_idx = {v: k for k, v in pos_idx_to_xy.items()}

  
for from_idx, (x, y) in pos_idx_to_xy.items():
    for a in range(4):  # actions 0-3
        new_x, new_y = x, y
        if a == 0: new_y -= 1  # up
        elif a == 1: new_y += 1  # down
        elif a == 2: new_x -= 1  # left
        elif a == 3: new_x += 1  # right

        new_pos = (new_x, new_y)
        if new_pos in xy_to_pos_idx:
            to_idx = xy_to_pos_idx[new_pos]
        else:
            to_idx = from_idx  # invalid move → stay in place

        B[0][to_idx, from_idx, a] = 1.0



# B[1] for doors_state_factor

# 0: both_closed -open-> red_open_blue_closed (0.5) or blue_open_red_closed (0.5)
# 1: blue_open_red_closed -open-> both_open (0.5) or blue_open_red_closed (0.5)
# 2: red_open_blue_closed -open-> both_open (0.5) or red_open_blue_closed (0.5)
# 3: both_open -open-> both_open (1.0)


# From state 0: both_closed
B[1][:, 0, 0] = 0
B[1][1, 0, 0] = 0.5  # → blue_open_red_closed
B[1][2, 0, 0] = 0.5  # → red_open_blue_closed

# From state 1: blue_open_red_closed
B[1][:, 1, 0] = 0
B[1][1, 1, 0] = 0.5  # stay in invalid state
B[1][3, 1, 0] = 0.5  # → both_open

# From state 2: red_open_blue_closed
B[1][:, 2, 0] = 0
B[1][2, 2, 0] = 0.5  # stay
B[1][3, 2, 0] = 0.5  # → both_open

# From state 3: both_open
B[1][:, 3, 0] = 0
B[1][3, 3, 0] = 1.0  # terminal


for f in range(num_factors):
    print(f"B[{f}] normalizes: {utils.is_normalized(B[f])}")


# C matrix      [for each state factor f: C[f][state_idx, state_idx]= P(state' | state)]
# reward logic: 
# -1: both_closed → no_progress
# -10: blue_open_red_closed → task_failed
# 0: red_open_blue_closed → progress_started
# 10: both_open → task_completed


outcome_obs_to_reward = { 
    "no_progress" : -1.0,      # both doors closed
    "task_failed" : -10.0,      # blue opened before red
    "progress_started" : 0.0, # red opened, blue not
    "task_completed" : 10.0,   # both open
}

C = utils.obj_array_zeros(num_obs)

# Set modality 0 (position) to neutral
C[0] = utils.norm_dist(softmax(np.zeros(num_obs[0])))  # shape: (9,)
# C[0] = np.zeros(num_obs[0]) # shape: (9,)

# Set modality 1 (outcome) based on reward and punishment
C[1] = utils.norm_dist(softmax(np.array([outcome_obs_to_reward['no_progress'], outcome_obs_to_reward["task_failed"], outcome_obs_to_reward["progress_started"], outcome_obs_to_reward["task_completed"]], dtype=np.float32)))
# C[1] = np.array([outcome_obs_to_reward['no_progress'], outcome_obs_to_reward["task_failed"], outcome_obs_to_reward["progress_started"], outcome_obs_to_reward["task_completed"]], dtype=np.float32)

# utils.plot_beliefs(softmax(C[1]), title = "Prior preferences")
for m in range(num_modalities):
    print(f"C[{m}] normalized: {utils.is_normalized(C[m])}")



# D matrix      [for each state factor f: D[f][state_idx]= P(state)]
D = utils.obj_array(num_factors)

# D[0]: Agent position → starts at (0,0) = "pos_0" = index 0
D[0] = np.zeros(num_states[0])
D[0][0] = 1.0

# D[1]: Door state → starts at "both_closed" = index 0
D[1] = np.zeros(num_states[1])
D[1][0] = 1.0

# utils.plot_beliefs(softmax(D[1]), title = "Prior beliefs about probability of the two contexts")
for f in range(num_factors):
    print(f"D[{f}] normalized: {utils.is_normalized(D[f])}")
    
    
    
# Initialize E matrix
E = utils.norm_dist(np.ones(shape=num_controls))
print(f"E normalized: {utils.is_normalized(E)}")

# priors over A for learning
pA = utils.dirichlet_like(A, scale = 1.0)

   
agent = Agent(A = A, B = B, C = C, D = D, E = E, pA = pA,
              inference_algo = 'MMP', policy_len=1,
              inference_horizon=2, sampling_mode = 'full',
              action_selection = 'stochastic')


# 1. Simulate an observation (e.g., sees pos_0 and no_progress)
obs = [0, 0]  # obs_position = pos_0, outcome = no_progress

# 2. Infer hidden states from observations
qs = agent.infer_states(obs)
# print("Free energy over policies (F):", agent.F)
# print("Posterior beliefs over states (qs):", qs)

# 3. Simulate choosing a hypothetical action (demo)
action_sampled_id = [0, 0]  # move 'up' and attempt to 'open'

# 4. Temporarily simulate expected next state given this action
agent.qs = copy.deepcopy(qs[action_sampled_id[0]][0])  # movement factor
agent.update_A(obs)  # learning: update pA based on obs
agent.qs = copy.deepcopy(qs)  # restore qs

# 5. Show updated likelihood
# print("Updated likelihood A[0]:", agent.A[0])
# print("Updated likelihood A[1]:", agent.A[1])

# 6. Policy inference
q_pi_raw, neg_efe = agent.infer_policies()
# print("Expected free energies (G):", -1 * agent.G)


# 7. Action selection

q_pi = np.mean(q_pi_raw, axis=1)  # shape → (4,)
q_pi = utils.norm_dist(np.nan_to_num(q_pi))


agent.q_pi = q_pi
action_sampled = agent.sample_action()


action_sampled = agent.sample_action()  # e.g., [2, 0]
action_sampled_id = [int(a) for a in action_sampled]
print("Sampled action:", action_sampled_id)

# 8. Policy posterior (q_pi)
q_pi = softmax(agent.G * agent.gamma - agent.F + np.log(agent.E))
print("Posterior over policies (q_pi):", q_pi)

# 9. Print policies (sequence of actions)
print("Candidate policies:", agent.policies)
