import numpy as np
import random
import json
import os

class QLearningAgent:
    def __init__(self, env, q_table_path=None, learning_rate=0.1, discount_factor=0.99, epsilon=1.0, epsilon_decay=0.995, min_epsilon=0.1):
        self.env = env
        self.actions = list(range(5))  # 6 possible actions (Up, Down, Left, Right, Stay, Open Door)
        
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.min_epsilon = min_epsilon
        self.q_table_path = q_table_path
        
        # Load Q-table or initialize new one
        # if q_table_path and os.path.exists(q_table_path):
        #     with open(q_table_path, 'r') as file:
        #         loaded_q_table = json.load(file)
        #     # Convert string keys back to tuples
        #     self.q_table = {eval(k): v for k, v in loaded_q_table.items()}
        # else:
        self.q_table = {}
    
    def get_state(self, obs):
        """Convert observations into a tuple state representation"""
        state = {}
        for agent, data in obs.items():
            state[agent] = (
                data['position'][0],  # X-coordinate
                data['position'][1],  # Y-coordinate
                data['red_door_opened'],
                data['blue_door_opened']
            )
        return tuple(state.items())
    
    def choose_action(self, state, agent_id):
        """Choose an action using an epsilon-greedy policy."""
        if random.random() < self.epsilon or state not in self.q_table:
            return random.choice(self.actions)
        else:
            return np.argmax(self.q_table[state][agent_id])
    
    def update_q_table(self, state, actions, rewards, next_state):
        """Update Q-table using the Q-learning update rule."""
        if state not in self.q_table:
            # self.q_table[state] = {agent: np.zeros(len(self.actions)) for agent in self.env.agents}
            self.q_table[state] = {agent: np.zeros(len(self.actions))}
            
        if next_state not in self.q_table:
            self.q_table[next_state] = {agent: np.zeros(len(self.actions))}
        
        for agent in self.env.agents:
            best_next_action = np.max(self.q_table[next_state][agent])
            self.q_table[state][agent][actions[agent]] = (
                (1 - self.learning_rate) * self.q_table[state][agent][actions[agent]]
                + self.learning_rate * (rewards[agent] + self.discount_factor * best_next_action)
            )
    
    def decay_exploration(self):
        """Reduce epsilon after each episode to encourage exploitation."""
        self.epsilon = max(self.min_epsilon, self.epsilon * self.epsilon_decay)
    
    def save_q_table(self):
        """Save the learned Q-table, converting tuple keys to strings."""
        if self.q_table_path:
            serializable_q_table = {str(k): v for k, v in self.q_table.items()}  # Convert keys to string
            
            with open(self.q_table_path, 'w') as file:
                json.dump(serializable_q_table, file)
            
            print(f"Q-table saved to {self.q_table_path}")