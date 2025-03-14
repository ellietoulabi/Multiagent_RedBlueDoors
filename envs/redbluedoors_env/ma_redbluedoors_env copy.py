import gym
from gym import spaces
import numpy as np
import json

class RedBlueDoorEnv(gym.Env):
    """ 
    Multi-Agent Red-Blue Door Environment (Ordinal Task)
    
    RedBlueDoors is an ordinal, two-agent, non-referential game. The environment consists of a red
    door and a blue door, both initially closed. The task of agents is to open both doors, but the
    order of actions matters. The red door must be opened first, followed by the blue door.
    
    - A reward of **1** is given to both agents if and only if the **red door is opened first** and then the **blue door**.
    - If the **blue door is opened first**, the **episode ends immediately** with a reward of **0** for both agents.
    - The blue agent must **wait** for the red agent to open the red door before successfully receiving a reward.
    - The task can be solved through **visual observation** or by a **single agent opening both doors**, meaning **explicit communication is not necessary**.
    
    """
    
    metadata = {'render.modes': ['human']}
    
    def __init__(self, config_path="configs/config.json"):
        super(RedBlueDoorEnv, self).__init__()

        # Load configuration
        with open(config_path, "r") as file:
            config = json.load(file)
        self.config = config
        self.width = config["width"]
        self.height = config["height"]
        self.red_door = tuple(config["red_door"])
        self.blue_door = tuple(config["blue_door"])
        self.walls = set(tuple(wall) for wall in config["walls"])

        self.num_agents = 2
        self.agent_positions = [tuple(pos) for pos in config["agent_positions"]]
        self.red_door_opened = False
        self.blue_door_opened = False

        # Define action space (0=Up, 1=Down, 2=Left, 3=Right, 4=Stay, 5=Open Door)
        self.action_space = spaces.Tuple(
            [spaces.Discrete(6) for _ in range(self.num_agents)]
        )

        # Observation: Agent positions, door states
        self.observation_space = spaces.Tuple(
            [
                spaces.Tuple(
                    [
                        spaces.Box(
                            low=0, high=self.width - 1, shape=(1,), dtype=np.int32
                        ),
                        spaces.Box(
                            low=0, high=self.height - 1, shape=(1,), dtype=np.int32
                        ),
                    ]
                )
                for _ in range(self.num_agents)
            ]
            + [spaces.Discrete(2), spaces.Discrete(2)]
        )

    def reset(self):
        """ Reset the environment to initial conditions """
        self.red_door_opened = False
        self.blue_door_opened = False
        self.agent_positions = [tuple(pos) for pos in self.config["agent_positions"]]
        return self._get_obs()

    def step(self, actions):
        """ Take a step for all agents """
        new_positions = list(self.agent_positions)
        rewards = [0, 0]
        dones = [False, False]

        proposed_positions = list(self.agent_positions)

        for i, action in enumerate(actions):
            x, y = self.agent_positions[i]
            new_x, new_y = x, y

            # Movement Actions
            if action == 0:  # Up
                new_y = max(0, y - 1)
            elif action == 1:  # Down
                new_y = min(self.height - 1, y + 1)
            elif action == 2:  # Left
                new_x = max(0, x - 1)
            elif action == 3:  # Right
                new_x = min(self.width - 1, x + 1)

            # Check if new position is valid and not causing a swap with another agent
            if (new_x, new_y) not in self.walls and \
            (new_x, new_y) not in proposed_positions and \
            (new_x, new_y) not in [self.red_door, self.blue_door]:
                proposed_positions[i] = (new_x, new_y)

        # Apply validated movement updates
        self.agent_positions = proposed_positions

        # Check door actions AFTER updating positions
        for i, action in enumerate(actions):
            x, y = self.agent_positions[i]  # Get updated position
            
            if action == 5:  # Open Door Action
                print(f"Agent {i} trying to open door at {x},{y}. Red: {self.red_door}, Blue: {self.blue_door}")
                
                if self._is_adjacent(x, y, self.blue_door) and not self.red_door_opened:
                    print(f"❌ Agent {i} attempted to open Blue Door before Red Door! Task failed.")
                    return self._get_obs(), [0, 0], [True, True], {}  # Episode ends with failure
                
                if self._is_adjacent(x, y, self.red_door) and not self.red_door_opened:
                    self.red_door_opened = True
                    rewards[i] = 10
                    print(f"✅ Agent {i} successfully opened Red Door!")
                elif self._is_adjacent(x, y, self.blue_door) and self.red_door_opened and not self.blue_door_opened:
                    self.blue_door_opened = True
                    rewards = [1, 1]  # Both agents get a reward of 1
                    dones = [True, True]
                    print(f"✅ Agent {i} successfully opened Blue Door! Task Completed Successfully.")

        return self._get_obs(), rewards, dones, {}
    
    def _is_adjacent(self, x, y, door):
        """ Check if (x, y) is adjacent to door """
        door_x, door_y = door
        return (abs(x - door_x) == 1 and y == door_y) or (abs(y - door_y) == 1 and x == door_x)
    
    def _get_obs(self):
        """ Returns the observation as a tuple of agent positions and door states """
        return tuple(self.agent_positions) + (int(self.red_door_opened), int(self.blue_door_opened))
    
    def render(self, mode='human'):
        """ Render the grid environment """
        grid = np.full((self.height, self.width), "_", dtype=str)
        
        # Place walls
        for wx, wy in self.walls:
            grid[wy, wx] = "#"
        
        # Place doors
        grid[self.red_door[1], self.red_door[0]] = "R" if not self.red_door_opened else "O"
        grid[self.blue_door[1], self.blue_door[0]] = "B" if not self.blue_door_opened else "O"
        
        # Place agents
        for i, (x, y) in enumerate(self.agent_positions):
            grid[y, x] = str(i)
        
        print("\n".join([" ".join(row) for row in grid]))
        print()

    def close(self):
        pass
