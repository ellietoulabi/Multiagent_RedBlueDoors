from pettingzoo.utils.env import ParallelEnv
import numpy as np
import json
from gym import spaces

# TODO: agents cannot occupy the same spot or move to a spot that is already occupied


class RedBlueSpotsEnv:

    def __init__(self, config_path="envs/redbluespots_env/configs/config.json"):
        super().__init__()

        self.episode_rewards = []
        self.current_reward = 0
        self.current_step = 0
        
        with open(config_path, "r") as file:
            config = json.load(file)
        self.config = config

        if "map" in config:
            self._parse_map(config["map"])
        else:
            self._parse_specs(config["specs"])

    def _parse_map(self, map_data):
        """Parses a textual map representation and extracts positions."""
        self.width = len(map_data[0].split())
        self.height = len(map_data)
        self.walls = set()
        self.agent_positions = {}
        self.spots = set()

        for y, row in enumerate(map_data):
            row = row.split()
            for x, char in enumerate(row):
                if char == "#":
                    self.walls.add((x, y))
                elif char == "S":
                    self.spots.add((x, y))
                elif char == "0":
                    self.agent_positions["agent_0"] = (x, y)
                elif char == "1":
                    self.agent_positions["agent_1"] = (x, y)

        self._initialize_common_attributes()
    def observe(self, agent):
      return self._get_obs(agent)
    def _parse_specs(self, config):
        """Parses specifications-based environment setup."""
        self.width = config["width"]
        self.height = config["height"]
        self.spots = set(tuple(spot) for spot in config["spots"])
        self.walls = set(tuple(wall) for wall in config["walls"])
        self.agent_positions = {
            "agent_0": tuple(config["agent_positions"][0]),
            "agent_1": tuple(config["agent_positions"][1]),
        }

        self._initialize_common_attributes()

    def _initialize_common_attributes(self):
        """Initialize shared attributes across both map and specs parsing."""
        self.agents = list(self.agent_positions.keys())
        self.actions = {0: "up", 1: "down", 2: "left", 3: "right", 4: "stay"}
        self.action_to_index = {v: k for k, v in self.actions.items()}
        self.spots_occupied = set()
        self.observation_spaces = {
            agent: spaces.Dict(
                {
                    "position": spaces.Box(
                        low=0,
                        high=max(self.width, self.height),
                        shape=(2,),
                        dtype=np.int32,
                    ),
                    "spots_occupied": spaces.Box(
                        low=-1,
                        high=max(self.width, self.height),
                        shape=(len(self.spots), 2),
                        dtype=np.int32,
                    ),
                }
            )
            for agent in self.agents
        }
        
        
        
    def seed(self, seed=None):
        np.random.seed(seed)
    
    def reset(self):
        """Resets the environment to its initial state."""
        self.agent_positions = self.config["agent_positions"]
        self.spots_occupied = set()
        self.episode_rewards.append(self.current_reward)
        self.current_reward = 0
        self.current_step = 0

        return self._get_observations()

    def _get_obs(self, agent):
        occupied = [pos for pos in self.agent_positions.values() if pos in self.spots]

        return {
            "position": np.array(self.agent_positions[agent], dtype=np.int32),
            "spots_occupied": np.array(occupied, dtype=np.int32),
        }

    def _get_observations(self):
        return {agent: self._get_obs(agent) for agent in self.agents}
    
    def _all_spots_occupied(self):
        return all(spot in self.agent_positions.values() for spot in self.spots)
        
    def _compute_spots_occupied(self):
        """Returns the set of currently occupied spots (agents standing on them)."""
        return set(pos for pos in self.agent_positions.values() if pos in self.spots)


    def step(self, actions):
        """Executes a step in the environment."""
        rewards = {agent: 0 for agent in self.agents}
        dones = {agent: False for agent in self.agents}
        infos = {agent: {} for agent in self.agents}

        for agent, action in actions.items():
            proposed_position = self._get_proposed_position(agent, action)
            if proposed_position not in self.walls:
                self.agent_positions[agent] = proposed_position

                 
        if self.spots.issubset(self._compute_spots_occupied()):
            for agent in self.agents:
                rewards[agent] = 10
                dones[agent] = True
        else:
            for agent in self.agents:
                rewards[agent] = -1

        self.current_reward += sum(rewards.values())
        self.current_step += 1  
        return self._get_observations(), rewards, dones, infos

    def _get_proposed_position(self, agent, action):
        """Returns the position the agent would move to given an action."""
        x, y = self.agent_positions[agent]
        if action == self.action_to_index['up']:
            return x, y - 1
        elif action == self.action_to_index['down']:
            return x , y + 1
        elif action == self.action_to_index['left']:
            return x - 1, y
        elif action == self.action_to_index['right']:
            return x + 1, y 
        elif action == self.action_to_index['stay']:
            return x, y 
        else:
            raise ValueError(f"Invalid action: {action}")

    

    def render(self, mode='human'):
        """Renders and returns the current grid state as a 2D array."""
        grid = np.full((self.height, self.width), "_", dtype=str)

        for wx, wy in self.walls:
            grid[wy, wx] = "#"

        

        for agent, (x, y) in self.agent_positions.items():
            grid[y, x] = agent[-1]  # '0' or '1'

        for sx, sy in self.spots:
            if grid[sy, sx] == "_":  # avoid overwriting occupied/agent
                grid[sy, sx] = "S"
        occupied = self._compute_spots_occupied()
        for ox, oy in occupied:
            grid[oy, ox] = "O"
        s = "\n".join(" ".join(row) for row in grid)
        print(s)
        print()
        return grid
    
    def close(self):
        pass  # For now, no resources to clean


    # def render(self):
    #     """Renders the current state of the environment."""
    #     occupied = self._compute_spots_occupied()
    #     for y in range(self.height):
    #         row = ""
    #         for x in range(self.width):
    #             if (x, y) in self.walls:
    #                 row += "# "
    #             elif (x, y) in occupied:
    #                 row += "O "
    #             elif (x, y) == self.agent_positions['agent_0']:
    #                 row += "0 "
    #             elif (x, y) == self.agent_positions['agent_1']:
    #                 row += "1 "
    #             elif (x, y) in self.spots:
    #                 row += "S "
    #             else:
    #                 row += "_ "
    #         print(row)
    #     print()