# from envs.redbluedoors_env.ma_redbluedoors_env import RedBlueDoorEnv  # Import your custom env
# from envs.redbluedoors_env.utils import visualize_trajectories
# import random

# env = RedBlueDoorEnv(config_path="envs/redbluedoors_env/configs/config.json")

# env.reset()
# env.render()
# trajectories = []
# num_steps = 10

# for step in range(num_steps):
#     # Actions: 0=Up, 1=Down, 2=Left, 3=Right, 4=Stay, 5=Open Door
#     actions = {"agent_0": random.randint(0, 5), "agent_1": random.randint(0, 5)}

#     obs, rewards, dones, _ = env.step(actions)
#     print(f"Step {step + 1}: Actions {actions}, Rewards {rewards}, Done {dones}")

#     s = env.render()

#     trajectories.append(s)
#     if dones['agent_0']==True and dones['agent_1']==True:  # Stop if the environment reaches a terminal state
#         print("Environment finished!")
#         break

# visualize_trajectories(trajectories)
# env.close()

from envs.redbluedoors_env.ma_redbluedoors_env import RedBlueDoorEnv
from agents.ql import QLearningAgent
from envs.redbluedoors_env.utils import visualize_trajectories
from envs.redbluedoors_env.ma_redbluedoors_env import RedBlueDoorEnv
from agents.ql import QLearningAgent
from agents.random import RandomAgent  
from utils.reward_plot import plot_cumulative_avg_rewards





# Configuration
CONFIG_PATH = "envs/redbluedoors_env/configs/config.json"
Q_TABLE_PATH = "q_table.json"
EPISODES = 10000  # Number of training episodes
MAX_STEPS = 50  # Maximum steps per episode

# Initialize environment
env = RedBlueDoorEnv(config_path=CONFIG_PATH)

# Initialize agents
ql_agent = QLearningAgent(env, Q_TABLE_PATH)
random_agent = RandomAgent(env)

# Track rewards
episode_rewards = []

# Training loop
for episode in range(EPISODES):
    obs, _ = env.reset()
    state = ql_agent.get_state(obs)
    total_reward = 0
    trajectories = []

    for step in range(MAX_STEPS):
        actions = {
            "agent_0": ql_agent.choose_action(state, "agent_0"),
            "agent_1": random_agent.choose_action(state, "agent_1"),
        }

        next_obs, rewards, dones, _ = env.step(actions)
        next_state = ql_agent.get_state(next_obs)

        ql_agent.update_q_table(state, actions, rewards, next_state)
        total_reward += sum(rewards.values())
        state = next_state

        # Save trajectory for visualization
        trajectories.append(env.render())

        if all(dones.values()):
            break

    # Store total reward per episode
    episode_rewards.append(total_reward)

    # Decay exploration rate
    ql_agent.decay_exploration()

    if episode % 500 == 0:
        print(
            f"Episode {episode}, Total Reward: {total_reward}, Epsilon: {ql_agent.epsilon:.4f}"
        )

# Save Q-table
# ql_agent.save_q_table()
print("Training complete. Q-table saved.")

# Close environment
env.close()



plot_cumulative_avg_rewards(episode_rewards)