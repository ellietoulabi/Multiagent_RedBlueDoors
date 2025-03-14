import unittest
import numpy as np
from envs.redbluedoors_env.ma_redbluedoors_env  import RedBlueDoorEnv


class TestRedBlueDoorEnv(unittest.TestCase):
    def setUp(self):
        """Initialize the environment before each test."""
        self.env = RedBlueDoorEnv(config_path="./envs/redbluedoors_env/configs/config.json")
        self.env.reset()
    
    def test_environment_initialization(self):
        """Test if the environment initializes correctly."""
        obs, _ = self.env.reset()
        self.assertEqual(len(obs), 2, "There should be 2 agents' observations.")
        for agent in self.env.agents:
            self.assertIn("position", obs[agent], "Each agent should have a position.")
            self.assertIn("red_door_opened", obs[agent], "Each agent should track red door state.")
            self.assertIn("blue_door_opened", obs[agent], "Each agent should track blue door state.")
    
    def test_agent_movement(self):
        """Test if agents move correctly within the grid without passing through walls or doors."""
        initial_positions = self.env.agent_positions.copy()
        actions = {"agent_0": 3, "agent_1": 3}  # Move both agents right
        obs, rewards, dones, infos = self.env.step(actions)
        new_positions = self.env.agent_positions
        
        for agent in self.env.agents:
            old_x, old_y = initial_positions[agent]
            new_x, new_y = new_positions[agent]
            self.assertTrue((new_x, new_y) not in self.env.walls, "Agent moved into a wall.")
            self.assertTrue((new_x, new_y) not in [self.env.red_door, self.env.blue_door], "Agent moved onto a door.")
    
    def test_open_red_door(self):
        """Test if an agent can open the red door when adjacent to it."""
        self.env.agent_positions["agent_0"] = (self.env.red_door[0] - 1, self.env.red_door[1])  # Position agent next to red door
        actions = {"agent_0": 5, "agent_1": 4}  # agent_0 tries to open door, agent_1 stays
        obs, rewards, dones, infos = self.env.step(actions)
        
        self.assertTrue(self.env.red_door_opened, "Red door should be opened.")
        self.assertEqual(rewards["agent_0"], 10, "Agent should receive 10 reward for opening red door.")
    
    def test_open_blue_door_without_red(self):
        """Test if an agent opening the blue door before the red door results in failure."""
        self.env.agent_positions["agent_1"] = (self.env.blue_door[0] - 1, self.env.blue_door[1])  # Position agent next to blue door
        actions = {"agent_0": 4, "agent_1": 5}  # agent_1 tries to open blue door first
        obs, rewards, dones, infos = self.env.step(actions)
        
        self.assertFalse(self.env.blue_door_opened, "Blue door should not be opened before red door.")
        self.assertTrue(all(dones.values()), "Episode should end if blue door is opened first.")
        self.assertEqual(rewards["agent_1"], 0, "Agent should receive 0 reward for opening blue door first.")
    
    def test_open_both_doors_in_order(self):
        """Test if agents get rewarded correctly when they open doors in the correct order."""
        # Position agent next to red door and open it
        self.env.agent_positions["agent_0"] = (self.env.red_door[0] - 1, self.env.red_door[1])
        actions = {"agent_0": 5, "agent_1": 4}
        obs, rewards, dones, infos = self.env.step(actions)
        self.assertTrue(self.env.red_door_opened, "Red door should be opened.")
        
        # Position agent next to blue door and open it
        self.env.agent_positions["agent_1"] = (self.env.blue_door[0] - 1, self.env.blue_door[1])
        actions = {"agent_0": 4, "agent_1": 5}
        obs, rewards, dones, infos = self.env.step(actions)
        
        self.assertTrue(self.env.blue_door_opened, "Blue door should be opened after red door.")
        self.assertTrue(all(dones.values()), "Episode should end when both doors are opened correctly.")
        self.assertEqual(rewards["agent_0"], 1, "Agent should receive reward for completing the task.")
        self.assertEqual(rewards["agent_1"], 1, "Agent should receive reward for completing the task.")
    
    def test_render(self):
        """Ensure the render function does not raise errors."""
        try:
            self.env.render()
        except Exception as e:
            self.fail(f"Render method raised an exception: {e}")

if __name__ == "__main__":
    unittest.main()