import gym
from gym.envs.registration import register

register(
    id='RedBlueDoor-v0',
    entry_point='envs.redbluedoors_env.ma_redbluedoors_env:RedBlueDoorEnv',
)
