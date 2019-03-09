# achtung-die-PLE

A try to master Achtung Die Kurve using Reinforcement learning.

Ended up creating a technically proficient little devil, 
but without any kind of tactical sense. This was due to
the environment the agent could see was narrowed down to 
a handful of beams, signalling the distance to a wall/opponent 
in a couple of different angles. 

`gym_achtung/envs` holds the `AchtungDieKurve` class,
which is basically the game, subclassed from OpenAI `gym.Env`.

`agent_achtung/TrainAchtung` are different training alternatives.
Full image uses the full image as input. Random Opponent competes against
random bots just thrown onto the playing field. There are also `EnjoyAchtung`
files which basically  

## Requirements

1. gym
2. baselines