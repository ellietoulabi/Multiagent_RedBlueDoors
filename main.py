from envs.redbluedoors_env.ma_redbluedoors_env import RedBlueDoorEnv  # Import your custom env
from envs.redbluedoors_env.utils import visualize_trajectories
import random

env = RedBlueDoorEnv(config_path="envs/redbluedoors_env/configs/config.json")

env.reset()
env.render()
trajectories = []
num_steps = 10

for step in range(num_steps):
    # Actions: 0=Up, 1=Down, 2=Left, 3=Right, 4=Stay, 5=Open Door
    actions = {"agent_0": random.randint(0, 5), "agent_1": random.randint(0, 5)}  
    
    obs, rewards, dones, _ = env.step(actions)
    print(f"Step {step + 1}: Actions {actions}, Rewards {rewards}, Done {dones}")
    
    s = env.render()
    
    trajectories.append(s)
    if dones['agent_0']==True and dones['agent_1']==True:  # Stop if the environment reaches a terminal state
        print("Environment finished!")
        break

visualize_trajectories(trajectories)
env.close()
