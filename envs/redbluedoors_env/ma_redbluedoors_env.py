from pettingzoo.utils.env import ParallelEnv
import numpy as np
import json
from gym import spaces

class RedBlueDoorEnv(ParallelEnv):
    """ 
    Multi-Agent Red-Blue Door Environment (Ordinal Task) using PettingZoo
    
    RedBlueDoors is an ordinal, two-agent, non-referential game. The environment consists of a red
    door and a blue door, both initially closed. The task of agents is to open both doors, but the
    order of actions matters. The red door must be opened first, followed by the blue door.
    
    - A reward of **1** is given to both agents if and only if the **red door is opened first** and then the **blue door**.
    - If the **blue door is opened first**, the **episode ends immediately** with a reward of **0** for both agents.
    - The task can be solved through **visual observation** or by a **single agent opening both doors**, meaning **explicit communication is not necessary**.
    
    """
    
    metadata = {"render.modes": ["human"]}

    def __init__(self, config_path="configs/config.json"):
        super().__init__()
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
        self.red_door = None
        self.blue_door = None

        for y, row in enumerate(map_data):
            row = row.split()
            for x, char in enumerate(row):
                if char == "#":
                    self.walls.add((x, y))
                elif char == "R":
                    self.red_door = (x, y)
                elif char == "B":
                    self.blue_door = (x, y)
                elif char == "0":
                    self.agent_positions["agent_0"] = (x, y)
                elif char == "1":
                    self.agent_positions["agent_1"] = (x, y)

        self._initialize_common_attributes()
    
    def _parse_specs(self, config):
        """Parses specifications-based environment setup."""
        self.width = config["width"]
        self.height = config["height"]
        self.red_door = tuple(config["red_door"])
        self.blue_door = tuple(config["blue_door"])
        self.walls = set(tuple(wall) for wall in config["walls"])
        self.agent_positions = {
            "agent_0": tuple(config["agent_positions"][0]),
            "agent_1": tuple(config["agent_positions"][1])
        }
        
        self._initialize_common_attributes()
    
    def _initialize_common_attributes(self):
        """Initialize shared attributes across both map and specs parsing."""
        self.agents = list(self.agent_positions.keys())
        self.red_door_opened = False
        self.blue_door_opened = False
        
        self.action_spaces = {agent: spaces.Discrete(5) for agent in self.agents}
        self.observation_spaces = {
            agent: spaces.Dict({
                "position": spaces.Box(low=0, high=max(self.width, self.height), shape=(2,), dtype=np.int32),
                "red_door_opened": spaces.Discrete(2),
                "blue_door_opened": spaces.Discrete(2)
            }) for agent in self.agents
        }

    def reset(self):
        self.red_door_opened = False
        self.blue_door_opened = False
       
        if "map" in self.config:
            self._parse_map(self.config["map"])
        else:
            self._parse_specs(self.config["specs"])
        
        return self._get_obs(), {}

    def step(self, actions):
        rewards = {agent: 0 for agent in self.agents}
        dones = {agent: False for agent in self.agents}
        infos = {agent: {} for agent in self.agents}

        # new_positions = self.agent_positions.copy()
        new_positions = dict(self.agent_positions)
        proposed_positions = {agent: new_positions[agent] for agent in self.agents}  


        for agent, action in actions.items():
            x, y = self.agent_positions[agent]
            new_x, new_y = x, y

            if action == 0:  # Up
                new_y = max(0, y - 1)
            elif action == 1:  # Down
                new_y = min(self.height - 1, y + 1)
            elif action == 2:  # Left
                new_x = max(0, x - 1)
            elif action == 3:  # Right
                new_x = min(self.width - 1, x + 1)

            # Only move if no swap happens
            if (new_x, new_y) not in self.walls and \
            (new_x, new_y) not in proposed_positions.values() and \
            (new_x, new_y) not in [self.red_door, self.blue_door]:
                proposed_positions[agent] = (new_x, new_y)
        # Apply movement after validating all
        self.agent_positions = proposed_positions
        
        for agent, action in actions.items():
            x, y = self.agent_positions[agent]
            
            if action == 4:  # Open Door Action
                if self._is_adjacent(x, y, self.blue_door) and not self.red_door_opened:
                    print(f"❌ {agent} attempted to open Blue Door before Red Door! Task failed.")
                    # return self._get_obs(), {agent: 0 for agent in self.agents}, {agent: True for agent in self.agents}, {}
                    dones = {agent: True for agent in self.agents}
                    return self._get_obs(), {agent: 0.0 for agent in self.agents}, dones, {}
                
                if self._is_adjacent(x, y, self.red_door) and not self.red_door_opened:
                    self.red_door_opened = True
                    # rewards[agent] = 10
                    print(f"✅ {agent} successfully opened Red Door!")
                elif self._is_adjacent(x, y, self.blue_door) and self.red_door_opened and not self.blue_door_opened:
                    self.blue_door_opened = True
                    rewards = {agent: 10 for agent in self.agents}
                    if self.blue_door_opened and self.red_door_opened:
                        dones = {agent: True for agent in self.agents}
                    else:
                        dones = {agent: False for agent in self.agents}
                    print(f"✅ {agent} successfully opened Blue Door! Task Completed Successfully.")

        return self._get_obs(), rewards, dones, infos

    def _get_obs(self):
        return {
            agent: {
                "position": np.array(self.agent_positions[agent], dtype=np.int32),
                "red_door_opened": int(self.red_door_opened),
                "blue_door_opened": int(self.blue_door_opened),
            } for agent in self.agents
        }

    def _is_adjacent(self, x, y, door):
        door_x, door_y = door
        return (abs(x - door_x) == 1 and y == door_y) or (abs(y - door_y) == 1 and x == door_x)
    
    def render(self, mode='human'):
        grid = np.full((self.height, self.width), "_", dtype=str)
        
        for wx, wy in self.walls:
            grid[wy, wx] = "#"
        
        grid[self.red_door[1], self.red_door[0]] = "R" if not self.red_door_opened else "O"
        grid[self.blue_door[1], self.blue_door[0]] = "B" if not self.blue_door_opened else "O"
        
        for agent, (x, y) in self.agent_positions.items():
            grid[y, x] = agent[-1]
        
        s = "\n".join([" ".join(row) for row in grid])
        print(s)
        print()
        # s = "".join("".join(row) + "\n" for row in grid)  # Ensure proper grid formatting

        return grid

    def close(self):
        pass
