from pymdp.agent import Agent as PymdpAgent
from pymdp import utils, maths
import random
import numpy as np


class ActiveInferenceAgent:
    def __init__(self, actions, A, B, C, D, policy_len, seed=42, control_fac_idx=None):
        self.internal_agent = PymdpAgent(
            A, B, C, D, control_fac_idx=control_fac_idx, policy_len=policy_len
        )
        super().__init__()
        self.all_actions = actions
        np.random.seed(seed)

    def reset(self):
        super().reset()
        self.internal_agent.reset()
        self.last_action = None
        self.last_state = None

    def _argmax_rand(self, a):
        indices = np.where(np.array(a) == np.max(a))[0]
        return np.random.choice(indices)

    @staticmethod
    def generate_matrix_A():

        states = {
            "self_pos": list(range(9)),
            "op_pos": list(range(9)),
        }

        observations = {
            "self_pos": list(range(9)),
            "op_pos": list(range(9)),
            "red_door": ["Close", "Open"],
            "blue_door": ["Close", "Open"],
        }
        state_shape = list(map(lambda x: len(x), states.values()))
        observation_shape = list(map(lambda x: len(x), observations.values()))
        A_shape = [[o_dim] + state_shape for o_dim in observation_shape]
        A = utils.obj_array_zeros(A_shape)
        # A[modality][obs,x state, y state, op x , op y , red, blue]
        # A[0][0,0,:,:,:,:,:] = 1.0
        # A[0][1,1,:,:,:,:,:] = 1.0
        # A[0][2,2,:,:,:,:,:] = 1.0

        # A[1][0,:,0,:,:,:,:] = 1.0
        # A[1][1,:,1,:,:,:,:] = 1.0
        # A[1][2,:,2,:,:,:,:] = 1.0

        # A[2][0,:,:,0,:,:,:] = 1.0
        # A[2][1,:,:,1,:,:,:] = 1.0
        # A[2][2,:,:,2,:,:,:] = 1.0

        # A[3][0,:,:,:,0,:,:] = 1.0
        # A[3][1,:,:,:,1,:,:] = 1.0
        # A[3][2,:,:,:,2,:,:] = 1.0

        # A[4][0,:,:,:,:,0,:] = 1.0
        # A[4][1,:,:,:,:,1,:] = 1.0

        # A[5][0,:,:,:,:,:,0] = 1.0
        # A[5][1,:,:,:,:,:,1] = 1.0

        # Modality 0: observe agent's x-position (depends on state factor 0)
        for i in range(9):
            A[0][i, i, :, :, :] = 1.0

        for i in range(9):
            A[1][i, :, i, :, :] = 1.0

        for i in range(2):
            A[2][i, :, :, i, :] = 1.0

        for i in range(2):
            A[3][i, :, :, i, :] = 1.0

        return A

    @staticmethod
    def generate_matrix_B():
        # states = {
        #     'self_x_pos': list(range(3)),
        #     'self_y_pos': list(range(3)),
        #     'op_x_pos': list(range(3)),
        #     'op_y_pos': list(range(3)),

        #     'red_door': ['Close', 'Open'],
        #     'blue_door': ['Close', 'Open'],
        # }
        states = {
            "self_pos": list(range(9)),
            "op_pos": list(range(9)),
            "red_door": ["Close", "Open"],
            "blue_door": ["Close", "Open"],
        }
        # ["up", "down", "left", "right", "open"]

        # controls = {
        #     'self_x_pos': ["up", "down", "left", "right", "open"],
        #     'self_y_pos': ["up", "down", "left", "right", "open"],
        #     'op_x_pos': ["up", "down", "left", "right", "open"],
        #     'op_y_pos': ["up", "down", "left", "right", "open"],

        #     'red_door': ["up", "down", "left", "right", "open"],
        #     'blue_door': ["up", "down", "left", "right", "open"],
        # }

        controls = {
            "self_pos": ["up", "down", "left", "right", "open"],
            "op_pos": ["up", "down", "left", "right", "open"],
            "red_door": ["up", "down", "left", "right", "open"],
            "blue_door": ["up", "down", "left", "right", "open"],
        }
        states_shape = list(map(lambda x: len(x), states.values()))
        controls_shape = list(map(lambda x: len(x), controls.values()))

        B_shape = [[ns, ns, controls_shape[f]] for f, ns in enumerate(states_shape)]
        B = utils.obj_array_zeros(B_shape)
        # self x
        # B[0][0,0,2] = 1.0
        # B[0][0,1,2] = 1.0
        # B[0][1,2,2] = 1.0

        # B[0][1,0,3] = 1.0
        # B[0][2,1,3] = 1.0
        # B[0][2,2,3] = 1.0

        # for i in range(3):
        #     B[0][i,i,0] = 1.0
        #     B[0][i,i,1] = 1.0
        #     B[0][i,i,4] = 1.0

        # self y
        # B[1][0,0,0] = 1.0
        # B[1][0,1,0] = 1.0
        # B[1][1,2,0] = 1.0

        # B[1][1,0,1] = 1.0
        # B[1][2,1,1] = 1.0
        # B[1][2,2,1] = 1.0

        # for i in range(3):
        #     B[1][i,i,2] = 1.0
        #     B[1][i,i,3] = 1.0
        #     B[1][i,i,4] = 1.0
        grid_dims = [3, 3]
        num_grid_points = np.prod(grid_dims)
        grid = np.arange(num_grid_points).reshape(grid_dims)
        it = np.nditer(grid, flags=["multi_index"])
        loc_list = []

        while not it.finished:
            loc_list.append(it.multi_index)
            it.iternext()

        actions = ["up", "down", "left", "right", "open"]
        for action_id, action_label in enumerate(actions):
            for curr_state, grid_location in enumerate(loc_list):

                y, x = grid_location

                if action_label == "up":
                    next_y = y - 1 if y > 0 else y
                    next_x = x
                elif action_label == "down":
                    next_y = y + 1 if y < (grid_dims[0] - 1) else y
                    next_x = x
                elif action_label == "left":
                    next_x = x - 1 if x > 0 else x
                    next_y = y
                elif action_label == "right":
                    next_x = x + 1 if x < (grid_dims[1] - 1) else x
                    next_y = y
                elif action_label == "open":
                    next_x = x
                    next_y = y

                new_location = (next_y, next_x)
                next_state = loc_list.index(new_location)
                B[0][next_state, curr_state, action_id] = 1.0
                B[1][next_state, curr_state, action_id] = 1.0

        #  # op x
        # B[2][0,0,2] = 1.0
        # B[2][0,1,2] = 1.0
        # B[2][1,2,2] = 1.0

        # B[2][1,0,3] = 1.0
        # B[2][2,1,3] = 1.0
        # B[2][2,2,3] = 1.0

        # for i in range(3):
        #     B[2][i,i,0] = 1.0
        #     B[2][i,i,1] = 1.0
        #     B[2][i,i,4] = 1.0

        # # op y
        # B[3][0,0,0] = 1.0
        # B[3][0,1,0] = 1.0
        # B[3][1,2,0] = 1.0

        # B[3][1,0,1] = 1.0
        # B[3][2,1,1] = 1.0
        # B[3][2,2,1] = 1.0

        # for i in range(3):
        #     B[3][i,i,2] = 1.0
        #     B[3][i,i,3] = 1.0
        #     B[3][i,i,4] = 1.0
        # red
        B[2][1, 0, 4] = 1.0
        B[2][1, 1, 4] = 1.0

        for i in range(2):
            B[2][i, i, 0] = 1.0
            B[2][i, i, 1] = 1.0
            B[2][i, i, 2] = 1.0
            B[2][i, i, 3] = 1.0

        # blue
        B[3][1, 0, 4] = 1.0
        B[3][1, 1, 4] = 1.0

        for i in range(2):
            B[3][i, i, 0] = 1.0
            B[3][i, i, 1] = 1.0
            B[3][i, i, 2] = 1.0
            B[3][i, i, 3] = 1.0

        return B

    @staticmethod
    def generate_matrix_C():
        # observations = {
        #     'self_x_pos': list(range(3)),
        #     'self_y_pos': list(range(3)),
        #     'op_x_pos': list(range(3)),
        #     'op_y_pos': list(range(3)),

        #     'red_door': ['Close', 'Open'],
        #     'blue_door': ['Close', 'Open'],
        # }
        observations = {
            "self_pos": list(range(9)),
            "op_pos": list(range(9)),
            "red_door": ["Close", "Open"],
            "blue_door": ["Close", "Open"],
        }
        observations_shape = list(map(lambda x: len(x), observations.values()))
        C = utils.obj_array_zeros(observations_shape)

        C[3][1] = 1

    @staticmethod
    def generate_matrix_D():
        # states = {
        #     'self_x_pos': list(range(3)),
        #     'self_y_pos': list(range(3)),
        #     'op_x_pos': list(range(3)),
        #     'op_y_pos': list(range(3)),

        #     'red_door': ['Close', 'Open'],
        #     'blue_door': ['Close', 'Open'],
        # }
        states = {
            "self_pos": list(range(9)),
            "op_pos": list(range(9)),
            "red_door": ["Close", "Open"],
            "blue_door": ["Close", "Open"],
        }
        states_shape = list(map(lambda x: len(x), states.values()))

        D = utils.obj_array_uniform(states_shape)

        D[0] = utils.onehot(0, 9)
        D[1] = utils.onehot(8, 9)
        D[2] = utils.onehot(0, 2)
        D[3] = utils.onehot(0, 2)

        return D

    def _get_observation(self, state):
        # if prev_action is None:
        #     return [0, 0, 0, 2, 0, 0]

        # self_x_pos = state["agent_0"]["position"][0]-1
        # self_y_pos = state["agent_0"]["position"][1]-1
        # op_x_pos = state["agent_1"]["position"][0]-1
        # op_y_pos = state["agent_1"]["position"][1]-1
        grid_dims = [3, 3]
        num_grid_points = np.prod(grid_dims)
        grid = np.arange(num_grid_points).reshape(grid_dims)
        it = np.nditer(grid, flags=["multi_index"])
        loc_list = []

        while not it.finished:
            loc_list.append(it.multi_index)
            it.iternext()

        self_pos = loc_list.index(
            (state["agent_0"]["position"][1] - 1, state["agent_0"]["position"][0] - 1)
        )
        op_pos = loc_list.index(
            (state["agent_1"]["position"][1] - 1, state["agent_1"]["position"][0] - 1)
        )

        red_door = state["agent_0"]["red_door_opened"]
        blue_door = state["agent_0"]["blue_door_opened"]

        # return [self_x_pos, self_y_pos, op_x_pos, op_y_pos, red_door, blue_door]
        return [self_pos, op_pos, red_door, blue_door]

    def choose_action(self, state, agent_id):
        obs = self._get_observation(state)

        self.internal_agent.infer_states(obs)
        self.internal_agent.infer_policies()
        action_id = self.internal_agent.sample_action()
        # print(action_id)
        if int(action_id[2]) == 4 or int(action_id[3]) == 4:
            action = 4
        else:
            action = int(action_id[0])
        self.last_action = action
        self.last_state = state
        # print("Action id",action_id)
        # print('choosed', Action.ACTION_TO_CHAR[action])

        return action
