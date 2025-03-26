import numpy as np
import matplotlib.pyplot as plt 


def plot_cumulative_avg_rewards(rewards):
    """Plots cumulative average reward over episodes."""
    avg_rewards = np.cumsum(rewards) / (np.arange(len(rewards)) + 1)

    plt.figure(figsize=(10, 5))
    plt.plot(avg_rewards, label="Cumulative Avg Reward", color="green")
    plt.xlabel("Episodes")
    plt.ylabel("Average Reward")
    plt.title("Cumulative Average Reward Over Episodes")
    plt.legend()
    plt.grid(True)
    plt.show()

