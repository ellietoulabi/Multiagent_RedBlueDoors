from envs.redbluespots_env.ma_redbluespots_env import RedBlueSpotsEnv
from envs.redbluespots_env import utils

env = RedBlueSpotsEnv()
trajectory = []
print("step 0")
trajectory.append(env.render())

print("step 1")
print(env.step({'agent_0':0, 'agent_1':0}))

trajectory.append(env.render())
print("step 2")
print(env.step({'agent_0':0, 'agent_1':0}))

trajectory.append(env.render())

print("step 3")
print(env.step({'agent_0':1, 'agent_1':1}))

trajectory.append(env.render())

print("step 4")
print(env.step({'agent_0':1, 'agent_1':0}))

trajectory.append(env.render())


utils.visualize_trajectories(trajectory)