import numpy as np
import random
import json
import os

class RandomAgent:
    def __init__(self, env):
        self.env = env
        self.actions = list(range(6))  # 6 possible actions
    
    def choose_action(self, _state, _agent_id):
        """Selects a random action from the action space."""
        return random.choice(self.actions)
